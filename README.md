# Pet Facial Recognition  

A high-performance facial recognition system designed for pet identification, built with Python, OpenCV, face_recognition library, and MongoDB. The system processes "wrong tag" frames, performs facial recognition, and stores results in a database with GridFS for image storage.

## Features

- **Real-time Facial Recognition**: Processes frames to identify pets using facial features
- **MongoDB Integration**: Stores metadata and images using GridFS
- **Batch Processing**: Efficiently handles large datasets with optimized resource management
- **Multi-threaded Processing**: Utilizes ThreadPoolExecutor for parallel processing
- **Configurable Settings**: Easy configuration through properties files
- **Excel Logging**: Generates detailed processing logs in Excel format

## Technology Stack

- **Python 3.6+**
- **OpenCV** - Image processing and computer vision
- **face_recognition** - Facial recognition library
- **MongoDB** - Database with GridFS for image storage
- **pymongo** - MongoDB Python driver
- **NumPy** - Numerical computations
- **Pandas** - Data processing and Excel export
- **Pillow** - Image handling

## Installation

### Prerequisites

1. **Python 3.6 or higher**
2. **MongoDB** installed and running locally or remotely
3. **Visual Studio Build Tools** (for Windows) - Required for compiling dependencies

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/pet_facial_recognition_mongodb.git
   cd pet_facial_recognition_mongodb
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   If you don't have a requirements.txt, install packages individually:
   ```bash
   pip install opencv-python==4.5.5.64 face-recognition pymongo pandas pillow numpy
   ```

## Configuration

1. **Create MongoDB configuration file** (`mongodb_config.properties`):
   ```properties
   mongo_uri=mongodb://localhost:27017/
   database_name=pet_facial_recognition_db
   encoding_threshold=0.6
   petdate=19-08-2025
   ```

2. **Prepare pickle files** containing registered pet data in the `singlebatch` folder:
   - Each pickle file should contain registration data with images
   - Images can be PIL Images, base64 strings, or numpy arrays

## Project Structure

```
pet_facial_recognition_mongodb/
├── mongodb_config.properties      # Database configuration
├── trynowfacial.py                # Main processing script
├── singlebatch/                   # Folder for registration pickle files
│   └── batch_1_images.pickle      # Example registration data
├── .venv/                         # Virtual environment (gitignored)
├── facial_logs_*.xlsx            # Generated log files (gitignored)
└── README.md                     # This file
```

## Usage

1. **Ensure MongoDB is running**
   ```bash
   mongod
   ```

2. **Run the facial recognition processor**
   ```bash
   python trynowfacial.py
   ```

3. **Monitor progress** - The script will:
   - Load registration data from pickle files
   - Find unprocessed frames in MongoDB
   - Process frames in batches with facial recognition
   - Update database with results
   - Generate Excel logs upon completion

## Database Schema to run in cmd
```cd C:\pet_facial_recognition_mongodb_wrongtag_livefacial```
```.venv\Scripts\activate.bat```
```python trynowfacial.py```


### Collections:
- `wrong_tag_results` - Stores frames awaiting processing
- `wrong_tag_images` - GridFS collection for storing images
- `bib_detection_results` - Stores successful recognition results

### Sample Document:
```json
{
  "bib_number": "0123",
  "rollno": "12345",
  "pet_date": "19-08-2025",
  "timestamp": 1629374825,
  "facial_flag": "1",
  "confidence": 0.85,
  "detection_method": "face_recognition",
  "image": ObjectId("...")
}
```

## Performance Optimization

The system includes several optimizations:
- **Batch processing** (50 frames per batch)
- **Thread-based parallelism** (4 workers max)
- **Resource monitoring** with delays between batches
- **Efficient memory management** with connection pooling

## Troubleshooting

### Common Issues:

1. **"ModuleNotFoundError: No module named 'cv2'"**
   ```bash
   pip install opencv-python==4.5.5.64
   ```

2. **"Insufficient system resources"**
   - Reduce batch size in code
   - Close other applications
   - Add more system RAM

3. **MongoDB connection issues**
   - Verify MongoDB is running
   - Check connection string in config file

4. **face_recognition installation failures**
   ```bash
   pip install cmake
   pip install dlib --only-binary=:all:
   pip install face-recognition
   ```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
 






