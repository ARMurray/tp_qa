# Diagnostics for the ICIS-NPDES (ICIS-COPY) pull. Read-only: it only runs SELECTs.
# Run from the correction/ folder (same as get_icis_npdes.R). Everything it writes
# is aggregate or permit-level (no secrets) and lands in data/icis_npdes/diagnostics/.
# Each step is wrapped so one failure does not stop the rest; check 00_run_log.txt.
#
# Credentials: set ICIS_USER and ICIS_PW in the R session first (e.g.
#   Sys.setenv(ICIS_USER = "...", ICIS_PW = "...")), or you will be prompted.
library(tidyverse)
library(DBI)
library(odbc)

# ---- connection (same as get_icis_npdes.R) ----------------------------------
my_user <- Sys.getenv("ICIS_USER")
my_pw   <- Sys.getenv("ICIS_PW")
if (my_user == "") my_user <- rstudioapi::askForPassword("ICIS-COPY user")
if (my_pw == "")   my_pw   <- rstudioapi::askForPassword("ICIS-COPY password")

Sys.setenv(TNS_ADMIN = "C:/ODBC/instantclient_23_0/network/admin")
Sys.setenv(PATH = paste("C:\\ODBC\\instantclient_23_0", Sys.getenv("PATH"), sep = ";"))
driver_name <- "{Oracle in instantclient_23_0}"
con <- dbConnect(odbc(),
                 .connection_string = sprintf("Driver=%s;Dbq=%s;Uid=%s;Pwd=%s;",
                                              driver_name, "ICISCOPY_SID", my_user, my_pw))

# ---- helpers -----------------------------------------------------------------
out_dir <- "data/icis_npdes/diagnostics"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
log_file <- file.path(out_dir, "00_run_log.txt")
cat("", file = log_file)
logmsg <- function(...) {
  msg <- paste(format(Sys.time(), "%H:%M:%S"), ...)
  cat(msg, "\n"); cat(msg, "\n", file = log_file, append = TRUE)
}
step <- function(name, expr) {
  logmsg("START", name)
  r <- tryCatch(expr, error = function(e) { logmsg("ERROR in", name, ":", conditionMessage(e)); NULL })
  logmsg("END  ", name)
  invisible(r)
}
save_csv <- function(df, name) write_csv(df, file.path(out_dir, name))
fmt_id <- function(x) if (is.character(x)) trimws(x) else sprintf("%.0f", as.numeric(x))
cutoff <- as.Date("2022-01-01")   # CWNS 2022 survey -> present

# Facility ownership from the ICIS facility type code (edit if you disagree)
pub_codes  <- c("CTG", "CNG", "STF", "MWD", "DIS", "SDT", "FDF", "TRB")
priv_codes <- c("POF", "COR", "IND", "NON")

# ---- 2. pull permits (all versions), facilities, components -------------------
step("2 pull", {
  perm <<- dbGetQuery(con, "SELECT ACTIVITY_ID, EXTERNAL_PERMIT_NMBR, VERSION_NMBR, PERMIT_TYPE_CODE, PERMIT_STATUS_CODE,
                            FACILITY_TYPE_INDICATOR, MAJOR_RATING_NMBR, MAJOR_MINOR_STATUS_FLAG, ORIGINAL_ISSUE_DATE,
                            ISSUE_DATE, EFFECTIVE_DATE, EXPIRATION_DATE, RETIREMENT_DATE, TERMINATION_DATE,
                            TOTAL_DESIGN_FLOW_NMBR, MASTER_EXTERNAL_PERMIT_NMBR, RECEIVING_POTW_ID, PERMIT_NAME
                            FROM ICIS_PERMIT")
  date_cols <- c("ORIGINAL_ISSUE_DATE", "ISSUE_DATE", "EFFECTIVE_DATE", "EXPIRATION_DATE", "RETIREMENT_DATE", "TERMINATION_DATE")
  perm <<- perm %>% mutate(ACTIVITY_ID = fmt_id(ACTIVITY_ID), VERSION_NMBR = as.integer(VERSION_NMBR),
                           across(all_of(date_cols), ~ as.Date(.x)))
  logmsg("permit rows:", nrow(perm), " distinct permits:", n_distinct(perm$EXTERNAL_PERMIT_NMBR))

  fac <<- dbGetQuery(con, "SELECT PGM_SYS_ID, FACILITY_TYPE_CODE, FACILITY_NAME, CITY, STATE_CODE, COUNTY_CODE, ZIP,
                           LOCATION_ADDRESS, FEDERAL_FACILITY_FLAG, TRIBAL_LAND_FLAG, HUC_CODE,
                           GEOCODE_LATITUDE, GEOCODE_LONGITUDE, GEOMETRIC_TYPE_CODE, HORIZONTAL_COLLECT_METHOD_CODE,
                           HORIZONTAL_REF_DATUM_CODE, REFERENCE_POINT_CODE, HORIZONTAL_ACCURACY_MEASURE,
                           SOURCE_MAP_SCALE_NMBR
                           FROM ICIS_FACILITY_INTEREST WHERE PGM_SYS_ACRNM = 'NPDES'")
  logmsg("facility rows:", nrow(fac), " distinct PGM_SYS_ID:", n_distinct(fac$PGM_SYS_ID))
  fac <<- fac %>% distinct(PGM_SYS_ID, .keep_all = TRUE)

  comp <<- dbGetQuery(con, "SELECT ACTIVITY_ID, COMPONENT_TYPE_CODE FROM XREF_PERM_COMPONENT_TYPE") %>%
    mutate(ACTIVITY_ID = fmt_id(ACTIVITY_ID))
  logmsg("component rows:", nrow(comp))
})

# ---- 3. one row per permit (lowest version) with flags ------------------------
step("3 permit table", {
  pot_act <- comp %>% filter(COMPONENT_TYPE_CODE == "POT") %>% distinct(ACTIVITY_ID) %>% mutate(pot = TRUE)
  perm <<- perm %>% left_join(pot_act, by = "ACTIVITY_ID") %>% mutate(pot = coalesce(pot, FALSE))
  per_permit <- perm %>% group_by(EXTERNAL_PERMIT_NMBR) %>%
    summarise(pot_any = any(pot), n_versions = n(), min_version = min(VERSION_NMBR, na.rm = TRUE),
              max_issue = suppressWarnings(max(ISSUE_DATE, na.rm = TRUE)), .groups = "drop")
  v0 <- perm %>% group_by(EXTERNAL_PERMIT_NMBR) %>% slice_min(VERSION_NMBR, n = 1, with_ties = FALSE) %>% ungroup() %>%
    left_join(per_permit, by = "EXTERNAL_PERMIT_NMBR") %>%
    left_join(fac, by = c("EXTERNAL_PERMIT_NMBR" = "PGM_SYS_ID")) %>%
    mutate(own = case_when(FACILITY_TYPE_CODE %in% pub_codes ~ "public",
                           FACILITY_TYPE_CODE %in% priv_codes ~ "private",
                           FACILITY_TYPE_CODE %in% c("MXO", "GOC") ~ "mixed",
                           is.na(FACILITY_TYPE_CODE) ~ "no_facility_row",
                           TRUE ~ "unknown"),
           alive = (is.na(TERMINATION_DATE) | TERMINATION_DATE >= cutoff) &
                   (is.na(RETIREMENT_DATE) | RETIREMENT_DATE >= cutoff) &
                   (PERMIT_STATUS_CODE %in% c("EFF", "ADC") | (!is.na(EXPIRATION_DATE) & EXPIRATION_DATE >= cutoff)),
           nm = toupper(coalesce(FACILITY_NAME, PERMIT_NAME, "")),
           sewer_name = grepl("WWTP|WWTF|WASTEWATER|WASTE WATER|SEWER|SEWAGE|WPCF|WPCP|\\bSTP\\b|POTW|TREATMENT (PLANT|FACILITY)|WATER POLLUTION|RECLAMATION|SANITARY", nm),
           uclass = case_when(
             PERMIT_TYPE_CODE == "NPD" & PERMIT_STATUS_CODE %in% c("EFF", "ADC") & pot ~ "A_original_NPD_EFF_ADC_POT",
             pot_any & PERMIT_TYPE_CODE %in% c("NPD", "GPC") & alive ~ paste0("B_POT_", PERMIT_TYPE_CODE, "_alive2022"),
             PERMIT_TYPE_CODE == "NPD" & alive & sewer_name ~ "C_noPOT_sewer_named_NPD_alive2022",
             TRUE ~ "Z_other"))
  v0 <<- v0
  logmsg("v0 rows:", nrow(v0))

  # permit census
  save_csv(v0 %>% count(PERMIT_TYPE_CODE, PERMIT_STATUS_CODE, own, pot, pot_any, alive, name = "n"), "02_permit_census.csv")
  save_csv(v0 %>% count(uclass, own, name = "n") %>% arrange(uclass, desc(n)), "02_universe_counts.csv")
  # version semantics
  vc <- tibble(
    metric = c("permits", "permits with no version 0", "permits where lowest version has the latest ISSUE_DATE",
               "permits with >1 version", "max versions on one permit", "ACTIVITY_ID duplicated across rows",
               "facility rows missing for permit (v0)"),
    value = c(nrow(v0), sum(v0$min_version != 0, na.rm = TRUE),
              sum(v0$ISSUE_DATE == v0$max_issue, na.rm = TRUE), sum(v0$n_versions > 1), max(v0$n_versions),
              sum(duplicated(perm$ACTIVITY_ID)), sum(v0$own == "no_facility_row")))
  save_csv(vc, "02_version_checks.csv")
  ex_ids <- v0 %>% filter(n_versions >= 3, pot_any) %>% slice_head(n = 25) %>% pull(EXTERNAL_PERMIT_NMBR)
  save_csv(perm %>% filter(EXTERNAL_PERMIT_NMBR %in% ex_ids) %>%
             select(EXTERNAL_PERMIT_NMBR, ACTIVITY_ID, VERSION_NMBR, PERMIT_TYPE_CODE, PERMIT_STATUS_CODE, ISSUE_DATE,
                    EFFECTIVE_DATE, EXPIRATION_DATE, RETIREMENT_DATE, TERMINATION_DATE, pot) %>%
             arrange(EXTERNAL_PERMIT_NMBR, VERSION_NMBR), "02_version_examples.csv")
  # when did POT permits that are no longer EFF/ADC end?
  save_csv(v0 %>% filter(pot_any, PERMIT_STATUS_CODE %in% c("EXP", "TRM", "ADC", "RET")) %>%
             mutate(end_year = as.integer(format(coalesce(TERMINATION_DATE, RETIREMENT_DATE, EXPIRATION_DATE), "%Y"))) %>%
             count(PERMIT_TYPE_CODE, PERMIT_STATUS_CODE, own, end_year, name = "n"), "02_end_year.csv")
  # how the original universe compares (should be ~13,682)
  logmsg("original-universe count (class A):", sum(v0$uclass == "A_original_NPD_EFF_ADC_POT"))
})

# ---- 4. is POT + ownership a usable "POTW" definition? ------------------------
step("4 POT definition", {
  save_csv(v0 %>% filter(alive) %>% count(FACILITY_TYPE_INDICATOR, pot, own, name = "n"), "03_indicator_crosstab.csv")
  save_csv(comp %>% inner_join(v0 %>% select(ACTIVITY_ID, own, alive, PERMIT_TYPE_CODE), by = "ACTIVITY_ID") %>%
             count(COMPONENT_TYPE_CODE, PERMIT_TYPE_CODE, own, alive, name = "n"), "03_component_census.csv")

  ops <- c("AMERICAN WATER" = "AMERICAN WATER|AMER WATER", "AQUA" = "\\bAQUA\\b", "SUEZ" = "SUEZ", "VEOLIA" = "VEOLIA",
           "UTILITIES INC" = "UTILITIES,? INC", "CORIX" = "CORIX", "LIBERTY" = "LIBERTY (UTIL|WATER)",
           "ESSENTIAL UTILITIES" = "ESSENTIAL UTIL", "CSWR" = "CENTRAL STATES WATER|CSWR", "UNITED WATER" = "UNITED WATER",
           "MIDDLESEX" = "MIDDLESEX WATER", "CAL WATER/GSWC" = "CALIFORNIA WATER|GOLDEN STATE WATER",
           "WATER/SEWER COMPANY" = "WATER (CO|COMPANY)\\b|SEWER (CO|COMPANY)\\b|UTILITY (CO|COMPANY)\\b",
           "UTILITY/UTILITIES" = "UTILIT",
           "MOBILE HOME/RV/TRAILER" = "MOBILE HOME|\\bMHP\\b|TRAILER|\\bRV\\b|CAMPGROUND",
           "HOA/SUBDIVISION/ESTATES" = "HOMEOWNER|\\bHOA\\b|SUBDIVISION|ESTATES|VILLAGE|MANOR|CONDO|APARTMENT",
           "SCHOOL/COLLEGE/PRISON/BASE" = "SCHOOL|COLLEGE|UNIVERSITY|PRISON|CORRECTION|\\bBASE\\b|AIR FORCE|ARMY|NAVY")
  priv_pot <- v0 %>% filter(pot_any, alive, own %in% c("private", "mixed", "unknown", "no_facility_row"))
  save_csv(map_dfr(names(ops), \(o) tibble(pattern = o, n_private_pot_alive = sum(grepl(ops[[o]], priv_pot$nm)),
                                           n_all_private_pot_alive = nrow(priv_pot))), "04_private_pot_operator_counts.csv")
  save_csv(priv_pot %>% arrange(desc(TOTAL_DESIGN_FLOW_NMBR)) %>% slice_head(n = 60) %>%
             select(EXTERNAL_PERMIT_NMBR, FACILITY_NAME, CITY, STATE_CODE, FACILITY_TYPE_CODE, PERMIT_TYPE_CODE,
                    PERMIT_STATUS_CODE, TOTAL_DESIGN_FLOW_NMBR, MAJOR_RATING_NMBR), "04_private_pot_top_by_flow.csv")
  # what does the POT tag miss? sewer-named NPD permits without POT
  nonpot <- v0 %>% filter(PERMIT_TYPE_CODE == "NPD", alive, sewer_name, !pot_any)
  save_csv(nonpot %>% count(own, FACILITY_TYPE_INDICATOR, name = "n"), "04_nonpot_sewer_named_counts.csv")
  save_csv(nonpot %>% slice_sample(n = min(300, nrow(nonpot))) %>%
             select(EXTERNAL_PERMIT_NMBR, FACILITY_NAME, CITY, STATE_CODE, FACILITY_TYPE_CODE, PERMIT_STATUS_CODE,
                    TOTAL_DESIGN_FLOW_NMBR, MAJOR_RATING_NMBR), "04_nonpot_sewer_named_sample.csv")
  # general permits with POT: which master permits?
  save_csv(v0 %>% filter(PERMIT_TYPE_CODE == "GPC", pot_any, alive) %>%
             count(STATE_CODE, MASTER_EXTERNAL_PERMIT_NMBR, own, name = "n") %>% arrange(desc(n)) %>% slice_head(n = 200),
           "05_gpc_masters.csv")
  # permit-level universe list (A, B, C) for joining to CWNS locally
  save_csv(v0 %>% filter(uclass != "Z_other") %>%
             select(EXTERNAL_PERMIT_NMBR, uclass, FACILITY_NAME, CITY, STATE_CODE, COUNTY_CODE, ZIP, LOCATION_ADDRESS,
                    FACILITY_TYPE_CODE, own, FACILITY_TYPE_INDICATOR, PERMIT_TYPE_CODE, PERMIT_STATUS_CODE, VERSION_NMBR,
                    n_versions, pot, pot_any, ISSUE_DATE, EFFECTIVE_DATE, EXPIRATION_DATE, TERMINATION_DATE,
                    TOTAL_DESIGN_FLOW_NMBR, MAJOR_RATING_NMBR, MASTER_EXTERNAL_PERMIT_NMBR, RECEIVING_POTW_ID,
                    HUC_CODE, ACTIVITY_ID), "06_permit_universe.csv")
})

# ---- 5. feature and coordinate coverage by universe class ---------------------
step("5 features", {
  # the full feature tables take ~30 minutes to pull, so cache them
  cache <- file.path(out_dir, "cache_features.rds")
  if (file.exists(cache)) {
    cc <- readRDS(cache); pf <- cc$pf; pfc <- cc$pfc
  } else {
    # only features on the lowest-version (VERSION_NMBR = 0) NPD/GPC permits, filtered on the server
    pf  <- dbGetQuery(con, "SELECT f.PERM_FEATURE_ID, f.ACTIVITY_ID, f.PERM_FEATURE_TYPE_CODE, f.PERM_FEATURE_NAME
                            FROM ICIS_PERM_FEATURE f
                            WHERE f.ACTIVITY_ID IN (SELECT ACTIVITY_ID FROM ICIS_PERMIT
                                                    WHERE VERSION_NMBR = 0 AND PERMIT_TYPE_CODE IN ('NPD', 'GPC'))")
    pfc <- dbGetQuery(con, "SELECT c.PERM_FEATURE_ID, c.LATITUDE_MEASURE, c.LONGITUDE_MEASURE, c.REFERENCE_POINT_CODE,
                            c.GEOMETRIC_TYPE_CODE, c.HORIZONTAL_COLLECT_METHOD_CODE, c.HORIZONTAL_REF_DATUM_CODE,
                            c.HORIZONTAL_ACCURACY_MEASURE
                            FROM ICIS_PERM_FEATURE_COORD c
                            WHERE c.PERM_FEATURE_ID IN (
                              SELECT f.PERM_FEATURE_ID FROM ICIS_PERM_FEATURE f
                              WHERE f.ACTIVITY_ID IN (SELECT ACTIVITY_ID FROM ICIS_PERMIT
                                                      WHERE VERSION_NMBR = 0 AND PERMIT_TYPE_CODE IN ('NPD', 'GPC')))")
    saveRDS(list(pf = pf, pfc = pfc), cache)   # large: do not commit cache_features.rds
  }
  pf  <- pf  %>% mutate(PERM_FEATURE_ID = fmt_id(PERM_FEATURE_ID), ACTIVITY_ID = fmt_id(ACTIVITY_ID))
  pfc <- pfc %>% mutate(PERM_FEATURE_ID = fmt_id(PERM_FEATURE_ID),
                        acc_bucket = cut(as.numeric(HORIZONTAL_ACCURACY_MEASURE), c(-Inf, 10, 30, 100, 1000, Inf),
                                         labels = c("<=10m", "<=30m", "<=100m", "<=1km", ">1km")))
  extra <- c("REFERENCE_POINT_CODE", "acc_bucket")

  fj <- pf %>% select(PERM_FEATURE_ID, ACTIVITY_ID, PERM_FEATURE_TYPE_CODE, PERM_FEATURE_NAME) %>%
    inner_join(perm %>% select(ACTIVITY_ID, EXTERNAL_PERMIT_NMBR, VERSION_NMBR), by = "ACTIVITY_ID") %>%
    left_join(v0 %>% select(EXTERNAL_PERMIT_NMBR, uclass, own), by = "EXTERNAL_PERMIT_NMBR") %>%
    left_join(pfc %>% select(PERM_FEATURE_ID, LATITUDE_MEASURE, LONGITUDE_MEASURE, GEOMETRIC_TYPE_CODE,
              HORIZONTAL_COLLECT_METHOD_CODE, HORIZONTAL_REF_DATUM_CODE, all_of(extra)), by = "PERM_FEATURE_ID") %>%
    mutate(has_coord = !is.na(LATITUDE_MEASURE) & !is.na(LONGITUDE_MEASURE))
  grp_cols <- c("uclass", "own", "PERM_FEATURE_TYPE_CODE", "GEOMETRIC_TYPE_CODE",
                "HORIZONTAL_COLLECT_METHOD_CODE", "HORIZONTAL_REF_DATUM_CODE", extra)
  save_csv(fj %>% filter(uclass != "Z_other") %>% group_by(across(any_of(grp_cols))) %>%
             summarise(features = n(), with_coord = sum(has_coord), .groups = "drop"), "07_feature_coord_summary.csv")
  # permits with at least one coordinate, by feature type
  cov <- fj %>% filter(uclass != "Z_other", has_coord) %>%
    group_by(uclass, own, EXTERNAL_PERMIT_NMBR) %>%
    summarise(any_coord = TRUE, inf = any(PERM_FEATURE_TYPE_CODE == "INF"), ino = any(PERM_FEATURE_TYPE_CODE == "INO"),
              exo = any(PERM_FEATURE_TYPE_CODE == "EXO"), .groups = "drop")
  save_csv(v0 %>% filter(uclass != "Z_other") %>% count(uclass, own, name = "permits") %>%
             left_join(cov %>% group_by(uclass, own) %>% summarise(with_any_coord = n(), with_inf = sum(inf),
                       with_ino = sum(ino), with_exo = sum(exo), .groups = "drop"), by = c("uclass", "own")),
           "07_permit_coord_coverage.csv")
  save_csv(fj %>% filter(uclass %in% c("A_original_NPD_EFF_ADC_POT", "B_POT_NPD_alive2022", "B_POT_GPC_alive2022"),
                         PERM_FEATURE_TYPE_CODE %in% c("INF", "INO", "EXO"), has_coord) %>%
             slice_sample(n = 600) %>% select(EXTERNAL_PERMIT_NMBR, uclass, PERM_FEATURE_TYPE_CODE, PERM_FEATURE_NAME,
                                             REFERENCE_POINT_CODE, HORIZONTAL_COLLECT_METHOD_CODE), "07_feature_names_sample.csv")
})

# ---- 6. facility-level coordinates (ICIS_FACILITY_INTEREST) by universe class --
step("6 facility coords", {
  fc <- v0 %>% filter(uclass != "Z_other") %>%
    mutate(has_geocode = !is.na(GEOCODE_LATITUDE) & !is.na(GEOCODE_LONGITUDE),
           lat = suppressWarnings(as.numeric(GEOCODE_LATITUDE)), lon = suppressWarnings(as.numeric(GEOCODE_LONGITUDE)),
           in_us_box = has_geocode & lat >= 17 & lat <= 72 & lon <= -60 & lon >= -180,
           acc_bucket = cut(as.numeric(HORIZONTAL_ACCURACY_MEASURE), c(-Inf, 10, 30, 100, 1000, Inf),
                            labels = c("<=10m", "<=30m", "<=100m", "<=1km", ">1km")))
  save_csv(fc %>% group_by(uclass, own, REFERENCE_POINT_CODE, HORIZONTAL_COLLECT_METHOD_CODE, HORIZONTAL_REF_DATUM_CODE,
                           acc_bucket) %>%
             summarise(permits = n(), with_geocode = sum(has_geocode), in_us_box = sum(in_us_box), .groups = "drop"),
           "08_facility_coord_summary.csv")
  save_csv(fc %>% group_by(uclass, own) %>%
             summarise(permits = n(), with_geocode = sum(has_geocode), with_huc = sum(!is.na(HUC_CODE)), .groups = "drop"),
           "08_facility_coord_coverage.csv")
  # pretreatment: IIU permits pointing at a receiving POTW
  iu <- perm %>% filter(!is.na(RECEIVING_POTW_ID))
  save_csv(tibble(metric = c("permit rows with RECEIVING_POTW_ID", "distinct receiving POTWs", "distinct IU permits"),
                  value = c(nrow(iu), n_distinct(iu$RECEIVING_POTW_ID), n_distinct(iu$EXTERNAL_PERMIT_NMBR))),
           "08_pretreatment_link_counts.csv")
})

dbDisconnect(con)
logmsg("DONE. Push the files in", out_dir)
