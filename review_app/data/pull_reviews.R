library(tidyverse)
library(arrow)
library(DBI)

con <- dbConnect(RSQLite::SQLite(), dbname = "review_app/data/app.db")

#dbListTables(con)
#candidates <- dbReadTable(con, "candidates")
#export_log <- dbReadTable(con,"export_log")

plants <- dbReadTable(con,"plants")


# Merge with existing best plants and update.
library(sf)

# List layers and find most recent
layers <- data.frame(layer = st_layers("C:/Users/AMURRA02/tp_qa/correction/data/training/Updates.gdb")$name)%>%
  separate(layer, into = c("A","B","date"), remove = FALSE)%>%
  select(layer,date)%>%
  mutate(date = mdy(date))%>%
  arrange(desc(date))

existing <- st_read("C:/Users/AMURRA02/tp_qa/correction/data/training/Updates.gdb", layer = layers$layer[1])%>%
  mutate(Verified = ifelse(is.na(Corrected),"No","Yes"))

# filter to already corrected
existing_corrected <- existing%>%
  filter(Corrected == "Yes")


# Get new decisions
plants_new <- plants%>%
  filter(!cwns_id %in% existing_corrected$CWNS_ID)%>%
  filter(plant_verdict %in% c("candidate_correct","reported_correct"))


# Get parcel centroids for new plants
states <- unique(plants_new$state_code)

# Iterate over states to pull parcels.



# Match formatting of existing
needed_cols <- existing%>%
  st_drop_geometry()%>%
  filter(CWNS_ID %in% plants_new$cwns_id)%>%
  select(!c(Original_Correct,How_Corrected,Verified,Corrected,Corrected_X,Corrected_Y))


plants_format <- plants_new%>%
  mutate(Original_Correct = ifelse(plant_verdict == "reported_correct","Yes","No"))%>%
  mutate(How_Corrected = ifelse(plant_verdict == "candidate_correct","Model",NA))%>%
  mutate(Corrected = ifelse(plant_verdict == "candidate_correct","Yes","No"))%>%
  mutate(Verified = "Yes",
        Corrected_X = ifelse(plant_verdict == "candidate_correct",longitude,NA),
        Corrected_Y = ifelse(plant_verdict == "candidate_correct",latitude,NA))%>%
  left_join(needed_cols, by = c("cwns_id"="CWNS_ID"))%>%
  select(cwns_id,INFRASTRUCTURE_TYPE,FACILITY_NAME,OWNER_TYPE,FACILITY_TYPE,LOCATION_TYPE,LATITUDE,LONGITUDE,ADDRESS,ADDRESS_2,CITY,STATE_CODE,ZIP_CODE,COUNTY_FIPS,COUNTY_NAME,Original_Correct,Original_X,Original_Y,Original_Source,Corrected_X,Corrected_Y,How_Corrected,Corrected,Verified)

colnames(plants_format)[1] <- "CWNS_ID"

# Re-combine
output <- existing%>%
  st_drop_geometry()%>%
  filter(!CWNS_ID %in% plants_format$CWNS_ID)%>%
  bind_rows(plants_format)%>%
  st_as_sf(coords = c("LONGITUDE","LATITUDE"),crs = st_crs(4326))

# Write new layer with date
layer_str <- paste0("CWNS_Locations_",format(Sys.Date(), "%m%d%Y"))

st_write(output,"C:/Users/AMURRA02/tp_qa/correction/data/training/Updates.gpkg", layer = layer_str)



crct <- read_parquet("C:/Users/AMURRA02/tp_qa/correction/diagnostics/output/cwns_corrected_locations.parquet")%>%
  filter(status %in% c("flagged_not_moved","moved","reviewed_unresolved","verified_corrected"))%>%
  select(!c("moved_to_ll_uuid", "reported_ll_uuid", "stage1_route", "stage1_prob_correct", "rerank_score", "rerank_fallback", "move_cutoff", "require_detection"))


write_csv(crct, "C:/Users/AMURRA02/tp_qa/correction/diagnostics/output/cwns_corrected_locations.csv")


# Histogram of moved distances
moved <- crct%>%
  filter(status %in% c("moved","verified_corrected"))

ggplot(moved, aes(x = moved_m/1000))+
  geom_histogram(bins = 50, color = "black", fill = "white")+
  scale_x_log10()+
  theme_bw()+
  labs(x = "Moved Distance (km)", y = "Count", title = "Histogram of Moved Distances for Corrected CWNS Locations",
subtitle = "1,667 Treatment Plants")

summary(moved$moved_m/1000)
