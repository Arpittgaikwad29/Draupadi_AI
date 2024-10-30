
# Draupadi-AI

Draupadi-AI is an AI-powered women's safety application designed to detect screams and specific keywords in real-time, sending alerts to designated authorities when threats are detected. The app aims to empower users by offering advanced security features, such as automated alert mechanisms, crisis reporting, location-based hotspot mapping, and a chatbot for additional support.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## Features

### 1. **Real-Time Scream and Keyword Detection**
   - The app continuously listens for specific keywords and screams.
   - When detected, alerts are sent to registered contacts or local authorities.

### 2. **False Positive Handling**
   - The app confirms anomalies by listening for a second trigger within one minute before sending alerts, reducing the risk of false alarms.
   - Users can cancel the alert or send it immediately.

### 3. **Map with Crime Hotspots**
   - Shows nearby crime hotspots categorized by severity.
   - Red indicates high-risk zones, orange indicates moderate risks, and green indicates low-risk areas.

### 4. **Sleep Mode**
   - A toggle button allows users to turn off the app’s audio detection for 30 minutes.
   - The app resumes detection automatically after 30 minutes, or users can manually resume detection sooner if desired.

### 5. **Emergency Button**
   - Users can save emergency contacts to send SMS alerts when a threat is detected.

### 6. **Safety News and Experience Sharing**
   - Share your experiences or safety insights about specific locations, helping other users stay informed.

### 7. **Chatbot for Crisis Reporting**
   - Users can communicate with a chatbot to report crises not detected by the model.
   - The chatbot triggers the alert mechanism and sends location-based alerts if needed.

## Installation

### Prerequisites
- Python 3.7+
- Flask
- TensorFlow or PyTorch (for model)
- Libraries: `librosa` (for audio processing)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Abhi951197/Draupadi-AI.git
   cd Draupadi-AI
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables for API keys (for map and SMS alerts).

4. Start the Flask server:
   ```bash
   python app.py
   ```

## Usage

1. **Open the Application**  
   Launch the app by accessing `localhost:5000` in your browser after starting the server.

2. **Setting Keywords**  
   Go to the settings modal and add a keyword you’d like the app to detect in addition to ‘help’.

3. **Activating Sleep Mode**  
   If you do not want the app to listen for audio, activate the sleep mode. After 30 minutes, it will automatically resume detection.

4. **Using the Chatbot**  
   To report a crisis not automatically detected, interact with the chatbot in the app to trigger an alert.

5. **Viewing Crime Hotspots**  
   Check the map feature to see crime hotspots in your area.

## API Endpoints

- `POST /set_keyword`: Sets a custom keyword for detection.
- `POST /trigger_alert`: Sends an alert if a scream or keyword is detected twice within one minute.
- `GET /get_hotspots`: Returns nearby crime hotspot locations.
- `POST /chatbot_alert`: Triggers an alert based on a manual report through the chatbot.

## Technologies Used
- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Flask, Python
- **Machine Learning**: TensorFlow/PyTorch for scream and keyword detection models
- **Database**: SQLite or MongoDB (for storing user data and logs)
- **APIs**: Map API for location-based services, SMS API for sending alerts

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch: `git checkout -b feature/your-feature`.
3. Commit your changes: `git commit -m 'Add new feature'`.
4. Push to the branch: `git push origin feature/your-feature`.
5. Submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
