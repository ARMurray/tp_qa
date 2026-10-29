library(tidyverse)
library(DBI)
library(odbc)
# Credentials
my_user <- "amurra02"
my_pw   <- "Fall_2026!"

# Ensure Instant Client path is on PATH for this session (optional if already set system-wide)
Sys.setenv(TNS_ADMIN = "C:/ODBC/instantclient_23_0/network/admin")
# Optional: ensure the Instant Client DLLs are on PATH for this session
Sys.setenv(PATH = paste("C:\\ODBC\\instantclient_23_0", Sys.getenv("PATH"), sep = ";"))

driver_name <- "{Oracle in instantclient_23_0}"

# If that fails with ORA-12514, try the SID alias:
con_sid <- dbConnect(odbc(),
                     .connection_string = sprintf("Driver=%s;Dbq=%s;Uid=%s;Pwd=%s;",
                                                  driver_name, "ICISCOPY_SID", my_user, my_pw))


# Identify permits with a POTW component type
comp <- dbGetQuery(con_sid,"SELECT * FROM XREF_PERM_COMPONENT_TYPE WHERE COMPONENT_TYPE_CODE = 'POT'")%>%
  mutate(ACTIVITY_ID = format(ACTIVITY_ID, scientific = FALSE, trim = TRUE),
         ACTIVITY_ID = as.character(ACTIVITY_ID))


# The universe of permits:
## NPDES permits that are version 0 (current) and have a permit type code of NPD, and are either:
## effective or administratively continued
permits <- dbGetQuery(con_sid,"SELECT ACTIVITY_ID, EXTERNAL_PERMIT_NMBR, PERMIT_TYPE_CODE,
ORIGINAL_ISSUE_DATE, ISSUE_DATE, EFFECTIVE_DATE, EXPIRATION_DATE, MAJOR_RATING_NMBR, TOTAL_DESIGN_FLOW_NMBR, ACTUAL_AVERAGE_FLOW_NMBR,
STATE_WATER_BODY, STATE_WATER_BODY_NAME, ISSUING_AGENCY, PERMIT_NAME FROM ICIS_PERMIT WHERE VERSION_NMBR=0 AND PERMIT_TYPE_CODE='NPD' AND (PERMIT_STATUS_CODE='EFF' OR PERMIT_STATUS_CODE='ADC')")%>%
  mutate(ACTIVITY_ID = format(ACTIVITY_ID, scientific = FALSE, trim = TRUE),
         ACTIVITY_ID = as.character(ACTIVITY_ID))%>%
  filter(ACTIVITY_ID %in% comp$ACTIVITY_ID)

# 45,214 -> 13,682

# Get facility locations for those permits
facilities <- dbGetQuery(con_sid,"SELECT ICIS_FACILITY_INTEREST_ID, PGM_SYS_ACRNM, PGM_SYS_ID, FACILITY_TYPE_CODE, FACILITY_NAME,
 LOCATION_ADDRESS, SUPPLEMENTAL_ADDRESS_TEXT, CITY, COUNTY_CODE, STATE_CODE, ZIP, FEDERAL_FACILITY_ID, FEDERAL_FACILITY_FLAG, TRIBAL_LAND_FLAG, FEDERAL_AGENCY_CODE
   FROM ICIS_FACILITY_INTEREST WHERE PGM_SYS_ACRNM='NPDES'")%>%
  filter(PGM_SYS_ID %in% permits$EXTERNAL_PERMIT_NMBR)

# Save the facilities table
write_csv(facilities,"data/icis_npdes/icis_npdes_facility_locations.csv") 


# Get permitted features for those permits
pf <- dbGetQuery(con_sid,"SELECT PERM_FEATURE_ID, DESIGN_FLOW_NMBR, OWNED_BY_FACILITY_FLAG, ACTUAL_AVERAGE_FLOW_NMBR, WATER_BODY_NAME, REACH_ID,
 PERM_FEATURE_NMBR, PERM_FEATURE_TYPE_CODE, STATE_WATER_BODY, STATE_WATER_BODY_NAME, ACTIVITY_ID FROM ICIS_PERM_FEATURE")%>%
  mutate(PERM_FEATURE_ID = format(PERM_FEATURE_ID, scientific = FALSE, trim = TRUE),
         PERM_FEATURE_ID = as.character(PERM_FEATURE_ID),
         ACTIVITY_ID = format(ACTIVITY_ID, scientific = FALSE, trim = TRUE),
         ACTIVITY_ID = as.character(ACTIVITY_ID))%>%
  filter(ACTIVITY_ID %in% comp$ACTIVITY_ID)

# Get coordinates for those permitted features
pf_coords <- dbGetQuery(con_sid,"SELECT PERM_FEATURE_ID, LATITUDE_MEASURE, LONGITUDE_MEASURE, GEOMETRIC_TYPE_CODE, 
HORIZONTAL_COLLECT_METHOD_CODE, HORIZONTAL_REF_DATUM_CODE FROM ICIS_PERM_FEATURE_COORD")%>%
  mutate(PERM_FEATURE_ID = format(PERM_FEATURE_ID, scientific = FALSE, trim = TRUE),
         PERM_FEATURE_ID = as.character(PERM_FEATURE_ID))%>%
  filter(PERM_FEATURE_ID %in% pf$PERM_FEATURE_ID)

# Join permitted features with coordinates
# Permit numbers
npdes_ids <- permits%>%
  select(ACTIVITY_ID, EXTERNAL_PERMIT_NMBR)

pf_join <- pf %>%
  left_join(pf_coords, by = "PERM_FEATURE_ID")%>%
  left_join(npdes_ids, by = "ACTIVITY_ID")

write_csv(pf_join,"data/icis_npdes/icis_npdes_permitted_features.csv")

geom <- dbGetQuery(con_sid,"SELECT * FROM REF_GEOMETRIC_TYPE")
write_csv(geom,"data/icis_npdes/icis_geometric_type.csv")

hm <- dbGetQuery(con_sid,"SELECT * FROM REF_HORIZONTAL_COLLECT_METHOD")
write_csv(hm,"data/icis_npdes/icis_horizontal_ref.csv")

datum_ref <- dbGetQuery(con_sid,"SELECT * FROM REF_HORIZONTAL_REF_DATUM")
write_csv(datum_ref,"data/icis_npdes/icis_horizontal_ref_datum.csv")

ref <- dbGetQuery(con_sid,"SELECT * FROM REF_REFERENCE_POINT")
write_csv(ref,"data/icis_npdes/icis_reference_point.csv")

uic <- dbGetQuery(con_sid,"SELECT * FROM REF_FAC_UNIVERSE_INDICATOR FETCH FIRST 1000 ROWS ONLY")
write_csv(uic,"data/icis_npdes/icis_fac_universe_indicator.csv")

# Facility Types
types <- dbGetQuery(con_sid,"SELECT * FROM REF_FACILITY_TYPE")
write_csv(types,"data/icis_npdes/icis_facility_type.csv")

dbDisconnect(con_sid)