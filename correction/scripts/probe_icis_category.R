# Read-only probe: where does the ECHO/ICIS dashboard's "Category" for a general permit live?
# Example from the dashboard: MAG580000, "DOMESTIC WASTEWATER TREATMENT FACILITIES".
# Run from the repo root or from correction/. Credentials as in get_icis_npdes.R.
# Output: data/icis_npdes/probe/*.csv + probe_log.txt  (small; push all of it)

library(tidyverse)
library(DBI)
library(odbc)

master_id <- "MAG580000"
phrase    <- "DOMESTIC WASTEWATER"

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

out_dir <- if (basename(getwd()) == "correction") "data/icis_npdes/probe" else "correction/data/icis_npdes/probe"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
log_file <- file.path(out_dir, "probe_log.txt")
cat("", file = log_file)
logmsg <- function(...) { m <- paste(format(Sys.time(), "%H:%M:%S"), ...); message(m); cat(m, "\n", file = log_file, append = TRUE) }

# run a query, save it, never stop the script
q <- function(name, sql) {
  t0 <- Sys.time()
  res <- tryCatch(dbGetQuery(con, sql), error = function(e) { logmsg(name, "FAILED:", conditionMessage(e)); NULL })
  if (!is.null(res)) {
    write_csv(res, file.path(out_dir, paste0(name, ".csv")), na = "")
    logmsg(name, nrow(res), "rows,", round(as.numeric(difftime(Sys.time(), t0, units = "secs"))), "s")
  }
  invisible(res)
}

# A. any column in the ICIS schema whose name looks like a category / sector / class
q("A_category_like_columns", "
  SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE, DATA_LENGTH FROM ALL_TAB_COLUMNS
  WHERE OWNER = 'ICIS' AND (COLUMN_NAME LIKE '%CATEG%' OR COLUMN_NAME LIKE '%PMOS%' OR COLUMN_NAME LIKE '%SECTOR%'
        OR COLUMN_NAME LIKE '%GENERAL%' OR COLUMN_NAME LIKE '%CLASS%' OR COLUMN_NAME LIKE '%PROGRAM%')
  ORDER BY TABLE_NAME, COLUMN_NAME")

# B. the general permit itself, all versions, and a count of its coverages
perm <- q("B_master_permit_rows", sprintf("
  SELECT ACTIVITY_ID, EXTERNAL_PERMIT_NMBR, VERSION_NMBR, PERMIT_TYPE_CODE, PERMIT_STATUS_CODE, PERMIT_NAME,
         FACILITY_TYPE_INDICATOR, PERM_INDUSTRIAL_CAT_CODE, MASTER_EXTERNAL_PERMIT_NMBR, ISSUING_AGENCY, COMMENT_TEXT
  FROM ICIS_PERMIT WHERE EXTERNAL_PERMIT_NMBR = '%s'", master_id))

cov <- q("B_coverage_rows", sprintf("
  SELECT ACTIVITY_ID, EXTERNAL_PERMIT_NMBR, VERSION_NMBR, PERMIT_TYPE_CODE, PERMIT_STATUS_CODE, PERMIT_NAME,
         FACILITY_TYPE_INDICATOR, PERM_INDUSTRIAL_CAT_CODE
  FROM ICIS_PERMIT WHERE MASTER_EXTERNAL_PERMIT_NMBR = '%s' AND VERSION_NMBR = 0
  ORDER BY EXTERNAL_PERMIT_NMBR", master_id))

# C. every feature column for the master permit's activities and for ten coverages
ids <- unique(c(perm$ACTIVITY_ID, head(cov$ACTIVITY_ID, 10)))
if (length(ids) > 0) {
  idl <- paste(ids, collapse = ",")
  q("C_features_all_columns", sprintf("SELECT * FROM ICIS_PERM_FEATURE WHERE ACTIVITY_ID IN (%s)", idl))
  q("C_components", sprintf("SELECT * FROM XREF_PERM_COMPONENT_TYPE WHERE ACTIVITY_ID IN (%s)", idl))
  q("C_industrial", sprintf("SELECT * FROM ICIS_PERM_INDUSTRIAL WHERE ACTIVITY_ID IN (%s)", idl))
  q("C_permit_all_columns", sprintf("SELECT * FROM ICIS_PERMIT WHERE ACTIVITY_ID IN (%s)", idl))
}

# D. distinct values of the leading candidate, with counts and a phrase test
q("D_master_general_category_values", "
  SELECT MASTER_GENERAL_CATEGORY_TEXT, PERM_FEATURE_TYPE_CODE, COUNT(*) N
  FROM ICIS_PERM_FEATURE WHERE MASTER_GENERAL_CATEGORY_TEXT IS NOT NULL
  GROUP BY MASTER_GENERAL_CATEGORY_TEXT, PERM_FEATURE_TYPE_CODE ORDER BY N DESC")

# E. reference tables that carry category names
q("E_ref_perm_industrial_cat", "SELECT * FROM REF_PERM_INDUSTRIAL_CAT")
q("E_ref_facility_type", "SELECT * FROM REF_FACILITY_TYPE")

# F. other tables that could hold the phrase on a permit or facility
q("F_phrase_in_permit_names", sprintf("
  SELECT EXTERNAL_PERMIT_NMBR, PERMIT_TYPE_CODE, PERMIT_NAME FROM ICIS_PERMIT
  WHERE VERSION_NMBR = 0 AND UPPER(PERMIT_NAME) LIKE '%%%s%%' AND ROWNUM <= 50", phrase))

dbDisconnect(con)
logmsg("DONE. Push", out_dir)
