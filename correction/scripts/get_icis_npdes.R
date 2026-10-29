# Pull ICIS-NPDES permits, facility locations, permitted features and pretreatment
# links from ICIS-COPY. Read-only (SELECT only). Run from the repo root or from correction/.
#
# Universe (decided after diag_icis_npdes.R, 2026-10-29):
#   * individual (NPD) and general-permit-coverage (GPC) permits, at their lowest
#     version number (version 0 = current), and
#   * alive since 2022-01-01 (not terminated or retired before, and either effective /
#     administratively continued or expiring in 2022 or later), and
#   * at least one of: a POT (POTW) component on any version, a sewer-type name, or SIC
#     4952 / NAICS 221320 (from sewerage_code_permits.csv, built from the ECHO download).
# The flags are kept as columns so the universe can be tightened downstream.
# Ownership is NOT classified here: FACILITY_TYPE_CODE is written as reported
# (it is unreliable, e.g. municipal plants coded COR).
#
# Outputs (data/icis_npdes/, NA written as empty):
#   icis_npdes_permits.csv       one row per permit: permit + facility fields + flags
#   icis_npdes_features.csv      permitted features of those permits, with coordinates
#   icis_npdes_pretreatment.csv  permits that name a receiving POTW (industrial users)
#   icis_npdes_pull_log.txt      counts for each step
#
# Credentials: set ICIS_USER / ICIS_PW in the R session (Sys.setenv), or you are prompted.
library(tidyverse)
library(DBI)
library(odbc)

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

out_dir <- if (basename(getwd()) == "correction") "data/icis_npdes" else "correction/data/icis_npdes"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
log_file <- file.path(out_dir, "icis_npdes_pull_log.txt")
cat("", file = log_file)
logmsg <- function(...) {
  msg <- paste(format(Sys.time(), "%H:%M:%S"), ...)
  cat(msg, "\n"); cat(msg, "\n", file = log_file, append = TRUE)
}
fmt_id <- function(x) if (is.character(x)) trimws(x) else sprintf("%.0f", as.numeric(x))
cutoff <- as.Date("2022-01-01")   # CWNS 2022 survey -> present
date_cols <- c("ORIGINAL_ISSUE_DATE", "ISSUE_DATE", "EFFECTIVE_DATE", "EXPIRATION_DATE", "RETIREMENT_DATE",
               "TERMINATION_DATE")

# ---- permits (all versions of NPD / GPC) ---------------------------------------
logmsg("pulling permits")
perm <- dbGetQuery(con, "SELECT ACTIVITY_ID, EXTERNAL_PERMIT_NMBR, VERSION_NMBR, PERMIT_TYPE_CODE, PERMIT_STATUS_CODE,
                         FACILITY_TYPE_INDICATOR, MAJOR_RATING_NMBR, MAJOR_MINOR_STATUS_FLAG, ORIGINAL_ISSUE_DATE,
                         ISSUE_DATE, EFFECTIVE_DATE, EXPIRATION_DATE, RETIREMENT_DATE, TERMINATION_DATE,
                         TOTAL_DESIGN_FLOW_NMBR, ACTUAL_AVERAGE_FLOW_NMBR, MASTER_EXTERNAL_PERMIT_NMBR,
                         ISSUING_AGENCY, AGENCY_TYPE_CODE, RECEIVING_POTW_ID, PERMIT_NAME
                         FROM ICIS_PERMIT WHERE PERMIT_TYPE_CODE IN ('NPD', 'GPC')") %>%
  mutate(ACTIVITY_ID = fmt_id(ACTIVITY_ID), VERSION_NMBR = as.integer(VERSION_NMBR),
         across(all_of(date_cols), as.Date))
logmsg("permit-version rows:", nrow(perm), " permits:", n_distinct(perm$EXTERNAL_PERMIT_NMBR))

# components (POT = POTW, PRE = pretreatment, CSO, ...) for every version
comp <- dbGetQuery(con, "SELECT ACTIVITY_ID, COMPONENT_TYPE_CODE FROM XREF_PERM_COMPONENT_TYPE") %>%
  mutate(ACTIVITY_ID = fmt_id(ACTIVITY_ID)) %>% distinct()
logmsg("component rows:", nrow(comp))

pot_act <- comp %>% filter(COMPONENT_TYPE_CODE == "POT") %>% distinct(ACTIVITY_ID) %>% mutate(pot = TRUE)
perm <- perm %>% left_join(pot_act, by = "ACTIVITY_ID") %>% mutate(pot = coalesce(pot, FALSE))
pot_any <- perm %>% group_by(EXTERNAL_PERMIT_NMBR) %>%
  summarise(pot_any_version = any(pot), n_versions = n(), .groups = "drop")
comp_str <- comp %>% group_by(ACTIVITY_ID) %>% summarise(components = paste(sort(COMPONENT_TYPE_CODE), collapse = ";"), .groups = "drop")

# lowest version per permit (version 0 = current; 502 permits had no version 0 in the diagnostics)
v0 <- perm %>% group_by(EXTERNAL_PERMIT_NMBR) %>% slice_min(VERSION_NMBR, n = 1, with_ties = FALSE) %>% ungroup() %>%
  left_join(pot_any, by = "EXTERNAL_PERMIT_NMBR") %>%
  left_join(comp_str, by = "ACTIVITY_ID") %>%
  rename(pot_lowest_version = pot) %>%
  mutate(alive_since_2022 = (is.na(TERMINATION_DATE) | TERMINATION_DATE >= cutoff) &
                            (is.na(RETIREMENT_DATE) | RETIREMENT_DATE >= cutoff) &
                            (PERMIT_STATUS_CODE %in% c("EFF", "ADC") | (!is.na(EXPIRATION_DATE) & EXPIRATION_DATE >= cutoff)))
logmsg("permits at lowest version:", nrow(v0), " alive since 2022:", sum(v0$alive_since_2022))

# ---- facilities (location, ownership code, coordinate quality) -----------------
logmsg("pulling facilities")
fac <- dbGetQuery(con, "SELECT PGM_SYS_ID, ICIS_FACILITY_INTEREST_ID, FACILITY_UIN, STATE_FACILITY_ID, FACILITY_TYPE_CODE,
                        FACILITY_NAME, LOCATION_ADDRESS, SUPPLEMENTAL_ADDRESS_TEXT, CITY, COUNTY_CODE, STATE_CODE, ZIP,
                        FEDERAL_FACILITY_FLAG, TRIBAL_LAND_FLAG, HUC_CODE,
                        GEOCODE_LATITUDE, GEOCODE_LONGITUDE, GEOMETRIC_TYPE_CODE, HORIZONTAL_COLLECT_METHOD_CODE,
                        HORIZONTAL_REF_DATUM_CODE, REFERENCE_POINT_CODE, HORIZONTAL_ACCURACY_MEASURE,
                        SOURCE_MAP_SCALE_NMBR
                        FROM ICIS_FACILITY_INTEREST WHERE PGM_SYS_ACRNM = 'NPDES'") %>%
  distinct(PGM_SYS_ID, .keep_all = TRUE)
logmsg("facility rows:", nrow(fac))

# ---- universe -------------------------------------------------------------------
sewer_rx <- paste0("WWTP|WWTF|WASTEWATER|WASTE WATER|SEWER|SEWAGE|WPCF|WPCP|\\bSTP\\b|POTW|",
                   "TREATMENT (PLANT|FACILITY)|WATER POLLUTION|RECLAMATION|SANITARY")
code_file <- file.path(out_dir, "sewerage_code_permits.csv")
if (file.exists(code_file)) {
  codes <- read_csv(code_file, col_types = "cii")
} else {
  logmsg("WARNING: sewerage_code_permits.csv not found; SIC/NAICS flags will be FALSE")
  codes <- tibble(EXTERNAL_PERMIT_NMBR = character(), sic_4952 = integer(), naics_221320 = integer())
}
permits <- v0 %>% filter(alive_since_2022) %>%
  left_join(fac, by = c("EXTERNAL_PERMIT_NMBR" = "PGM_SYS_ID")) %>%
  left_join(codes, by = "EXTERNAL_PERMIT_NMBR") %>%
  mutate(sic_4952 = coalesce(sic_4952, 0L) == 1L, naics_221320 = coalesce(naics_221320, 0L) == 1L,
         sewerage_code = sic_4952 | naics_221320,
         sewer_named = grepl(sewer_rx, toupper(coalesce(FACILITY_NAME, PERMIT_NAME, ""))),
         has_pre_component = grepl("PRE", coalesce(components, "")),
         has_cso_component = grepl("CSO", coalesce(components, ""))) %>%
  filter(pot_any_version | sewer_named | sewerage_code)
logmsg("universe permits:", nrow(permits), " POT any version:", sum(permits$pot_any_version),
       " sewer-named only:", sum(permits$sewer_named & !permits$pot_any_version & !permits$sewerage_code),
       " sewerage-code only:", sum(permits$sewerage_code & !permits$pot_any_version & !permits$sewer_named),
       " with facility geocode:", sum(!is.na(permits$GEOCODE_LATITUDE)))
write_csv(permits, file.path(out_dir, "icis_npdes_permits.csv"), na = "")

# ---- permitted features with coordinates ----------------------------------------
# Server-side filter to lowest-version NPD/GPC activities, then trimmed to the universe in R.
# (Permits with no version 0 get no features here; the count is logged.)
logmsg("pulling features")
act_sql <- "SELECT ACTIVITY_ID FROM ICIS_PERMIT WHERE VERSION_NMBR = 0 AND PERMIT_TYPE_CODE IN ('NPD', 'GPC')"
pf <- dbGetQuery(con, sprintf("SELECT PERM_FEATURE_ID, ACTIVITY_ID, PERM_FEATURE_NMBR, PERM_FEATURE_TYPE_CODE,
                               PERM_FEATURE_NAME, WATER_BODY_NAME, REACH_ID, DESIGN_FLOW_NMBR, OWNED_BY_FACILITY_FLAG
                               FROM ICIS_PERM_FEATURE WHERE ACTIVITY_ID IN (%s)", act_sql)) %>%
  mutate(PERM_FEATURE_ID = fmt_id(PERM_FEATURE_ID), ACTIVITY_ID = fmt_id(ACTIVITY_ID))
pfc <- dbGetQuery(con, sprintf("SELECT c.PERM_FEATURE_ID, c.LATITUDE_MEASURE, c.LONGITUDE_MEASURE, c.GEOMETRIC_TYPE_CODE,
                                c.HORIZONTAL_COLLECT_METHOD_CODE, c.HORIZONTAL_REF_DATUM_CODE, c.REFERENCE_POINT_CODE,
                                c.HORIZONTAL_ACCURACY_MEASURE, c.SOURCE_MAP_SCALE_NMBR
                                FROM ICIS_PERM_FEATURE_COORD c
                                WHERE c.PERM_FEATURE_ID IN (SELECT f.PERM_FEATURE_ID FROM ICIS_PERM_FEATURE f
                                                            WHERE f.ACTIVITY_ID IN (%s))", act_sql)) %>%
  mutate(PERM_FEATURE_ID = fmt_id(PERM_FEATURE_ID))
features <- pf %>%
  inner_join(permits %>% select(ACTIVITY_ID, EXTERNAL_PERMIT_NMBR), by = "ACTIVITY_ID") %>%
  left_join(pfc, by = "PERM_FEATURE_ID") %>%
  relocate(EXTERNAL_PERMIT_NMBR)
logmsg("features:", nrow(features), " with coordinates:", sum(!is.na(features$LATITUDE_MEASURE)),
       " universe permits with no features pulled:", sum(!permits$EXTERNAL_PERMIT_NMBR %in% features$EXTERNAL_PERMIT_NMBR))
write_csv(features, file.path(out_dir, "icis_npdes_features.csv"), na = "")

# ---- pretreatment: permits that name a receiving POTW -----------------------------
logmsg("pulling pretreatment links")
iu <- dbGetQuery(con, "SELECT ACTIVITY_ID, EXTERNAL_PERMIT_NMBR, VERSION_NMBR, PERMIT_TYPE_CODE, PERMIT_STATUS_CODE,
                       PERMIT_NAME, ISSUE_DATE, EFFECTIVE_DATE, EXPIRATION_DATE, RETIREMENT_DATE, TERMINATION_DATE,
                       RECEIVING_POTW_ID, IU_SIGNIFICANT_FLAG, CONTROL_AUTH_PERMIT_ID, PERM_INDUSTRIAL_CAT_CODE
                       FROM ICIS_PERMIT WHERE RECEIVING_POTW_ID IS NOT NULL") %>%
  mutate(ACTIVITY_ID = fmt_id(ACTIVITY_ID), VERSION_NMBR = as.integer(VERSION_NMBR),
         across(any_of(date_cols), as.Date)) %>%
  group_by(EXTERNAL_PERMIT_NMBR) %>% slice_min(VERSION_NMBR, n = 1, with_ties = FALSE) %>% ungroup() %>%
  left_join(fac %>% select(PGM_SYS_ID, FACILITY_NAME, LOCATION_ADDRESS, CITY, STATE_CODE, ZIP, GEOCODE_LATITUDE,
                           GEOCODE_LONGITUDE, HORIZONTAL_COLLECT_METHOD_CODE, HORIZONTAL_REF_DATUM_CODE,
                           REFERENCE_POINT_CODE, HORIZONTAL_ACCURACY_MEASURE),
            by = c("EXTERNAL_PERMIT_NMBR" = "PGM_SYS_ID"))
logmsg("pretreatment permits:", nrow(iu), " receiving POTWs:", n_distinct(iu$RECEIVING_POTW_ID),
       " with facility geocode:", sum(!is.na(iu$GEOCODE_LATITUDE)))
write_csv(iu, file.path(out_dir, "icis_npdes_pretreatment.csv"), na = "")

dbDisconnect(con)
logmsg("DONE. Push the files in", out_dir)
