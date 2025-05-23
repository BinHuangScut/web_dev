from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torchvision.transforms as transforms
import io
import os
import uuid # For generating unique filenames
from datetime import datetime # For timestamps in filenames or records
from fastapi.staticfiles import StaticFiles # Added for static files
from fastapi.responses import FileResponse # Added for serving index.html

from sqlalchemy.orm import Session # Added for DB session
from . import crud, models, schemas # Added for DB operations
from .database import SessionLocal, engine, create_db_and_tables, get_db # Added

# Determine the correct path to the frontend build directory
# This assumes 'backend' and 'frontend' are sibling directories in the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FRONTEND_BUILD_DIR = os.path.join(BASE_DIR, "frontend", "build")

# Create database and tables on startup
# In a more complex app, you might use Alembic for migrations
# and manage table creation outside the app server startup.
create_db_and_tables() 

app = FastAPI()

# CORS (Cross-Origin Resource Sharing) middleware
# Allow all origins for development, restrict in production
origins = [
    "http://localhost",
    "http://localhost:3000", # Assuming React frontend runs on port 3000
    # Add your deployed frontend URL here for production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- File Upload Directory ---
# Determine the base directory of the backend
# __file__ is backend/app/main.py, so os.path.dirname(__file__) is backend/app
# os.path.dirname(os.path.dirname(__file__)) is backend/
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True) # Create uploads directory if it doesn't exist

# --- Model Definition ---
model_name = "SqueezeNet 1.1"
try:
    # Try to load from a specific torchvision version if needed, e.g., 'pytorch/vision:v0.13.0'
    # Using 'pytorch/vision' generally tries to pick a compatible version.
    model = torch.hub.load('pytorch/vision', 'squeezenet1_1', pretrained=True)
    model.eval()  # Set to evaluation mode
    print(f"{model_name} loaded successfully and set to evaluation mode.")
except Exception as e:
    print(f"Error loading model {model_name}: {e}")
    model = None # Ensure model is None if loading fails

# --- Image Preprocessing ---
# Standard preprocessing for ImageNet models
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Load Class Categories ---
categories = []
try:
    # Determine the script's directory to find imagenet_classes.txt
    script_dir = os.path.dirname(__file__)
    # Construct the full path to the imagenet_classes.txt file
    # It should be in the 'backend' directory, so one level up from 'app'
    classes_file_path = os.path.join(script_dir, "..", "imagenet_classes.txt")
    
    with open(classes_file_path, "r") as f:
        categories = [s.strip() for s in f.readlines()]
    print(f"Loaded {len(categories)} categories from {classes_file_path}")
except FileNotFoundError:
    print(f"Error: imagenet_classes.txt not found at {classes_file_path}. Predictions will be class indices.")
    categories = [str(i) for i in range(1000)] # Fallback
except Exception as e:
    print(f"An error occurred while loading categories: {e}")
    categories = [str(i) for i in range(1000)] # Fallback


@app.get("/")
def read_root():
    return {"message": f"Image Classification API with {model_name if model else 'No Model Loaded'}"}

@app.post("/api/predict", response_model=schemas.PredictionResponse)
async def predict_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check server logs.")
    if not categories:
        raise HTTPException(status_code=503, detail="ImageNet categories not loaded. Please check server logs.")

    file_path = None # Initialize file_path to None
    try:
        # 1. Read image
        contents = await file.read()
        input_image = Image.open(io.BytesIO(contents)).convert("RGB")

        # --- Save the uploaded file ---
        # Generate a unique filename to prevent overwrites and add timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        unique_id = uuid.uuid4().hex[:6] # Short unique ID
        # Sanitize filename (optional, but good practice)
        original_filename = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in file.filename)
        saved_filename = f"{timestamp}_{unique_id}_{original_filename}"
        file_path = os.path.join(UPLOAD_DIR, saved_filename)

        # Write the file (contents were already read, so re-open in binary write)
        # It's better to save the file first before processing
        with open(file_path, "wb") as f:
            f.write(contents)
        # --- End file saving ---

        # 2. Preprocess image (using the already opened input_image)
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

        # 3. Perform inference (on CPU)
        with torch.no_grad():  # Important: disable gradient calculation
            output = model(input_batch)

        # 4. Get probabilities and top prediction
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        top1_prob, top1_catid = torch.topk(probabilities, 1)
        
        predicted_class_idx = top1_catid[0].item()
        
        # Ensure the index is within the bounds of the categories list
        if 0 <= predicted_class_idx < len(categories):
            predicted_class_name = categories[predicted_class_idx]
        else:
            predicted_class_name = f"Unknown class index: {predicted_class_idx}"
            print(f"Warning: Predicted class index {predicted_class_idx} is out of bounds for {len(categories)} categories.")

        confidence = top1_prob[0].item()

        # --- Save prediction to database ---
        prediction_data = schemas.PredictionCreate(
            original_filename=file.filename,
            saved_filename=saved_filename,
            file_path=file_path, # Store the server-side path
            predicted_class=predicted_class_name,
            confidence=confidence,
            model_version=model_name,
            prediction_time=datetime.utcnow() # Explicitly set prediction time
        )
        db_prediction_record = crud.create_prediction_record(db=db, prediction=prediction_data)
        # --- End database saving ---

        # The Pydantic response_model will automatically convert db_prediction_record
        return db_prediction_record
        
    except HTTPException as http_exc: # Re-raise HTTPExceptions
        raise http_exc
    except Exception as e:
        print(f"Error during prediction for {file.filename}: {e}")
        # Clean up saved file if prediction fails after saving
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Cleaned up partially saved file: {file_path}")
            except OSError as ose:
                print(f"Error cleaning up file {file_path}: {ose}")
        return HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
    finally:
        if file: # Ensure file object exists before trying to close
            await file.close()

# --- Mount static files (for React build) ---
# This should be AFTER your API routes
app.mount("/static", StaticFiles(directory=os.path.join(FRONTEND_BUILD_DIR, "static")), name="static")

@app.get("/{full_path:path}")
async def serve_react_app(full_path: str):
    """Serves the React app's index.html for any route not handled by API or static files."""
    index_path = os.path.join(FRONTEND_BUILD_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        # This case should ideally not be hit if the frontend is built correctly
        raise HTTPException(status_code=404, detail="Frontend index.html not found. Ensure the frontend is built.")

# To run this app:
# 1. Navigate to the 'image-classifier/backend' directory in your terminal
# 2. Ensure you have an 'app' subdirectory with this 'main.py' file inside it.
# 3. Ensure 'imagenet_classes.txt' is in the 'backend' directory (one level above 'app').
# 4. An 'uploads' directory will be automatically created in the 'backend' directory.
# 5. Make sure your PostgreSQL server is running and you have created a database
#    (e.g., 'image_classifier_db') that your DATABASE_URL in database.py points to.
# 6. Run: uvicorn app.main:app --reload
#    The --reload flag is for development and auto-reloads on code changes. 