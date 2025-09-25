Frontend Development Notes
These notes are a detailed guide for building the frontend based on the Figma designs and integrating it with the backend API.

General API Information
Base URL: The API's base URL will be provided after deployment to Render/Railway (e.g., https://my-attendance-api.onrender.com). All API calls should be prefixed with this URL.

Authentication: Endpoints for administrative tasks (enrollment, manual attendance, etc.) require a secure X-API-Key in the request headers. The frontend should have a form field for the teacher to enter this key, and the key should be stored securely (e.g., in localStorage).

Response Structure:

Success: A JSON object with "status": "success" and a "data" field containing the requested information.

Error: A JSON object with "status": "error", an "error_type" (e.g., "AuthenticationError", "ValidationError"), and a user-friendly "message".

Page-by-Page API Integration
1. Dashboard 
This page requires the frontend to fetch summary data and display it in cards.

API Call: GET /api/attendance/summary

Purpose: To populate the "Total Users," "Present Today," and "Total Absent" cards.

Data: The API returns a JSON object with these counts. The frontend should calculate the "Attendance Rate" as a percentage (Present / Total).

Authentication: Not required.

API Call: GET /api/attendance/export?date=YYYY-MM-DD

Purpose: To trigger a CSV file download when the "Export" button is clicked.

Data: The date parameter should be the selected date from the UI.

Authentication: Not required.

Feature Flags: Call GET /api/features on page load to determine which attendance methods are available. If face_recognition or ocr is false, the "Picture Attendance" and "Scan RFID Sheet" cards should be disabled or visually indicate that the feature is not available on the server.

2. Manual Attendance 
This page is for a teacher to manually update attendance.

API Call: GET /api/students/names

Purpose: To get a list of all student names to build the attendance table.

Authentication: Requires X-API-Key.

API Call: GET /api/get_attendance?date=YYYY-MM-DD

Purpose: To pre-populate the checkboxes based on today's attendance status.

Data: The date parameter should be the current date. The response contains a dictionary of students and their statuses. The frontend should set the checkbox to checked if student.status is "Present".

Authentication: Not required.

API Call: POST /api/manual_attendance

Purpose: To save manual changes.

Request Body: The frontend should build a JSON object with the date and an array of student updates.

Authentication: Requires X-API-Key.

Example Request:

{
  "date": "2025-09-26",
  "updates": [
    { "name": "John Doe", "status": "Present" },
    { "name": "Jane Smith", "status": "Absent" }
  ]
}

3. RFID Attendance 
This page handles RFID card scanning.

API Call: POST /api/rfid_scan

Purpose: To mark a student present based on their RFID tag.

Data: The frontend should send a form-encoded value for rfid_tag.

Authentication: Not required.

4. Scan RFID Sheet 
This page processes a photo of an attendance sheet using OCR.

API Call: POST /api/ocr_process

Purpose: To process an uploaded image and update attendance.

Data: The frontend should send a multipart/form-data request with the image file.

Authentication: Requires X-API-Key.

5. Picture Attendance 
This page processes photos for face recognition.

API Call: POST /api/process

Purpose: To process one or more uploaded images and update attendance.

Data: The frontend should send a multipart/form-data request with the photo files.

Authentication: Requires X-API-Key.
