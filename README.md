
# Flow of the application

Process CSV --> detect_departments_from_sop --> resolve_initial_department --> call_validate_region_tool  --> Complaint Summarizer --> Finalize Json Builder Ticket


# Executing the code
docker-compose build --no-cache
docker-compose up


