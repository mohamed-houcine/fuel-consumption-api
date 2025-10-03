# main.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Body, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import joblib
import pandas as pd
import os
import io
import json
import re
from enum import Enum

app = FastAPI(title="Fuel Consumption Prediction API ")

# CORS (dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.isdir("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

# ---- Features exposées ----
NUM_FEATS = ['city08', 'highway08', 'displ', 'cylinders']
CAT_FEATS = [
    'drive', 'fuelType', 'fuelType1', 'fuelType2',
    'model_grouped', 'make_grouped', 'VClass_grouped'
]
ALL_FEATS = NUM_FEATS + CAT_FEATS

# ------- Pydantic model -------
class FuelInput(BaseModel):
    city08: Optional[float] = Field(None, description="City MPG")
    highway08: Optional[float] = Field(None, description="Highway MPG")
    displ: Optional[float] = Field(None, description="Engine displacement (L)")
    cylinders: Optional[int] = Field(None, description="Number of cylinders")

    drive: Optional[str] = Field(None, description="Drive type (e.g., FWD, RWD, AWD)")
    fuelType: Optional[str] = Field(None, description="Fuel type (grouped)")
    fuelType1: Optional[str] = Field(None, description="Fuel type1 (if available)")
    fuelType2: Optional[str] = Field(None, description="Fuel type2 (if available)")
    model_grouped: Optional[str] = Field(None, description="Model grouped (top N or 'Other')")
    make_grouped: Optional[str] = Field(None, description="Make grouped (top N or 'Other')")
    VClass_grouped: Optional[str] = Field(None, description="Vehicle class grouped")

    class Config:
        schema_extra = {
            "example": {
                "city08": 20.0,
                "highway08": 30.0,
                "displ": 2.0,
                "cylinders": 4,
                "drive": "FWD",
                "fuelType": "Regular Gasoline",
                "fuelType1": None,
                "fuelType2": None,
                "model_grouped": "Corolla",
                "make_grouped": "Toyota",
                "VClass_grouped": "Compact Cars"
            }
        }
        extra = "allow"


# ------- Feature enum (string enum for Swagger) -------
class FeatureName(str, Enum):
    city08 = "city08"
    highway08 = "highway08"
    displ = "displ"
    cylinders = "cylinders"
    drive = "drive"
    fuelType = "fuelType"
    fuelType1 = "fuelType1"
    fuelType2 = "fuelType2"
    model_grouped = "model_grouped"
    make_grouped = "make_grouped"
    VClass_grouped = "VClass_grouped"


# ------- Enums for categorical values -------
class VClassGrouped(str, Enum):
    two_seaters = "Two Seaters"
    subcompact_cars = "Subcompact Cars"
    other = "Other"
    compact_cars = "Compact Cars"
    midsize_cars = "Midsize Cars"
    large_cars = "Large Cars"
    small_station_wagons = "Small Station Wagons"
    standard_pickup_trucks = "Standard Pickup Trucks"
    sport_utility_vehicle_4wd = "Sport Utility Vehicle - 4WD"
    small_sport_utility_vehicle_4wd = "Small Sport Utility Vehicle 4WD"
    standard_sport_utility_vehicle_4wd = "Standard Sport Utility Vehicle 4WD"


class MakeGrouped(str, Enum):
    other = "Other"
    dodge = "Dodge"
    toyota = "Toyota"
    volkswagen = "Volkswagen"
    bmw = "BMW"
    chevrolet = "Chevrolet"
    nissan = "Nissan"
    ford = "Ford"
    mercedes_benz = "Mercedes-Benz"
    gmc = "GMC"
    porsche = "Porsche"


class ModelGrouped(str, Enum):
    other = "Other"
    truck_2wd = "Truck 2WD"
    f150_pickup_2wd = "F150 Pickup 2WD"
    ranger_pickup_2wd = "Ranger Pickup 2WD"
    sierra_1500_2wd = "Sierra 1500 2WD"
    f150_pickup_4wd = "F150 Pickup 4WD"
    camaro = "Camaro"
    mustang = "Mustang"
    civic = "Civic"
    accord = "Accord"
    jetta = "Jetta"


class FuelType(str, Enum):
    regular = "Regular"
    premium = "Premium"
    diesel = "Diesel"
    cng = "CNG"
    gasoline_or_natural_gas = "Gasoline or natural gas"
    gasoline_or_e85 = "Gasoline or E85"
    electricity = "Electricity"
    gasoline_or_propane = "Gasoline or propane"
    premium_or_e85 = "Premium or E85"
    midgrade = "Midgrade"
    premium_gas_or_electricity = "Premium Gas or Electricity"
    regular_gas_and_electricity = "Regular Gas and Electricity"
    premium_and_electricity = "Premium and Electricity"
    regular_gas_or_electricity = "Regular Gas or Electricity"
    hydrogen = "Hydrogen"


class FuelType1(str, Enum):
    regular_gasoline = "Regular Gasoline"
    premium_gasoline = "Premium Gasoline"
    diesel = "Diesel"
    natural_gas = "Natural Gas"
    electricity = "Electricity"
    midgrade_gasoline = "Midgrade Gasoline"
    hydrogen = "Hydrogen"


class FuelType2(str, Enum):
    missing = "missing"
    natural_gas = "Natural Gas"
    e85 = "E85"
    propane = "Propane"
    electricity = "Electricity"


class DriveType(str, Enum):
    rear_wheel_drive = "Rear-Wheel Drive"
    front_wheel_drive = "Front-Wheel Drive"
    four_wheel_or_all_wheel_drive = "4-Wheel or All-Wheel Drive"
    missing = "missing"
    two_wheel_drive = "2-Wheel Drive"
    all_wheel_drive = "All-Wheel Drive"
    four_wheel_drive = "4-Wheel Drive"
    part_time_four_wheel_drive = "Part-time 4-Wheel Drive"


# ------- Enhanced Pydantic model for selection -------
class VehicleSelection(BaseModel):
    city08: Optional[float] = Field(None, description="City MPG", ge=1, le=150)
    highway08: Optional[float] = Field(None, description="Highway MPG", ge=1, le=150)
    displ: Optional[float] = Field(None, description="Engine displacement (L)", ge=0.5, le=10.0)
    cylinders: Optional[int] = Field(None, description="Number of cylinders", ge=2, le=16)

    drive: Optional[DriveType] = Field(None, description="Drive type")
    fuelType: Optional[FuelType] = Field(None, description="Fuel type")
    fuelType1: Optional[FuelType1] = Field(None, description="Primary fuel type")
    fuelType2: Optional[FuelType2] = Field(None, description="Secondary fuel type")
    model_grouped: Optional[ModelGrouped] = Field(None, description="Vehicle model (grouped)")
    make_grouped: Optional[MakeGrouped] = Field(None, description="Vehicle manufacturer (grouped)")
    VClass_grouped: Optional[VClassGrouped] = Field(None, description="Vehicle class (grouped)")

    class Config:
        schema_extra = {
            "example": {
                "city08": 25.0,
                "highway08": 35.0,
                "displ": 2.5,
                "cylinders": 4,
                "drive": "Front-Wheel Drive",
                "fuelType": "Regular",
                "fuelType1": "Regular Gasoline",
                "fuelType2": "missing",
                "model_grouped": "Civic",
                "make_grouped": "Toyota",
                "VClass_grouped": "Compact Cars"
            }
        }


# ------- Load model/pipeline & metadata -------
pipeline = None
model = None
feature_names = None
medians = None

if os.path.exists("rf_pipeline_v1.joblib"):
    pipeline = joblib.load("rf_pipeline_v1.joblib")
    print("Loaded: rf_pipeline_v1.joblib (pipeline)")
else:
    if os.path.exists("rf_tuned.joblib"):
        model = joblib.load("rf_tuned.joblib")
        print("Loaded: rf_tuned.joblib (model)")
    else:
        raise RuntimeError("No model found. Provide rf_pipeline_v1.joblib or rf_tuned.joblib in the working dir.")

    if os.path.exists("feature_names.joblib"):
        feature_names = joblib.load("feature_names.joblib")
    if os.path.exists("medians.joblib"):
        medians = joblib.load("medians.joblib")


# ------- helper predict function -------
def _predict_from_df(df: pd.DataFrame) -> List[float]:
    if pipeline is not None:
        preds = pipeline.predict(df)
        return preds.tolist()

    df_dummies = pd.get_dummies(df)
    if feature_names is not None:
        df_aligned = df_dummies.reindex(columns=feature_names, fill_value=0)
    else:
        df_aligned = df_dummies.copy()

    if medians is not None:
        for col, val in medians.items():
            if col in df_aligned.columns:
                df_aligned[col] = df_aligned[col].fillna(val)

    preds = model.predict(df_aligned)
    return preds.tolist()


# ------- Improved parsers and normalizer -------
def _try_parse_number(s: str):
    if s is None:
        return None
    s = s.strip()
    if s == "" or s.lower() in ("none", "null"):
        return None
    # try integer then float
    try:
        if re.fullmatch(r"-?\d+", s):
            return int(s)
        if re.fullmatch(r"-?\d+\.\d+", s):
            f = float(s)
            return int(f) if f.is_integer() else f
    except Exception:
        pass
    # fallback: return original string
    return s


def parse_text_to_rows(text: str) -> List[Dict[str, Any]]:
    """
    Robust parser for text files. Supports:
     - JSON object or JSON list of objects
     - CSV (only if it looks like a CSV i.e. has commas/tabs/semicolons in header)
     - key=value or key: value lines, blocks separated by blank lines (one record per block)
     - also supports multiple key=value pairs on one line separated by whitespace/commas/semicolons
    """
    text = text.strip()
    if not text:
        return []

    # Try JSON first
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return [obj]
        if isinstance(obj, list) and all(isinstance(x, dict) for x in obj):
            return obj
    except Exception:
        pass

    # Heuristic: treat as CSV only if the first non-empty line contains a comma, tab or semicolon
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 1:
        first = lines[0]
        if ("," in first) or ("\t" in first) or (";" in first):
            try:
                df = pd.read_csv(io.StringIO(text))
                if not df.empty:
                    return df.to_dict(orient="records")
            except Exception:
                # If CSV parsing fails, continue to fallback parsing
                pass

    # Fallback: parse key=value / key: value blocks
    rows = []
    # split blocks on one or more blank lines
    blocks = re.split(r'\n\s*\n+', text)
    for b in blocks:
        b = b.strip()
        if not b:
            continue
        row: Dict[str, Any] = {}
        for line in b.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # If the line contains multiple pairs separated by commas/semicolons/whitespace, split tokens
            tokens = re.split(r'[;,]\s*', line)
            # if we didn't split (no commas/semicolons), tokens will be [line]
            for token in tokens:
                token = token.strip()
                if not token:
                    continue
                # if token contains whitespace separated pairs like "a=1 b=2", split by whitespace and process each
                if re.search(r'\s', token) and ('=' in token or ':' in token):
                    subtoks = re.split(r'\s+', token)
                else:
                    subtoks = [token]
                for st in subtoks:
                    if not st:
                        continue
                    if "=" in st:
                        k, v = st.split("=", 1)
                    elif ":" in st:
                        k, v = st.split(":", 1)
                    else:
                        # unknown token, skip
                        continue
                    k = k.strip()
                    v = v.strip()
                    val = _try_parse_number(v)
                    row[k] = val
        if row:
            rows.append(row)
    return rows


def normalize_rows(raw_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    normalized = []
    for r in raw_rows:
        row = {f: None for f in ALL_FEATS}
        for k, v in r.items():
            if k in ALL_FEATS:
                if v is None:
                    row[k] = None
                elif isinstance(v, str) and v.strip() == "":
                    row[k] = None
                else:
                    row[k] = v
            else:
                # ignore unknown keys
                pass
        for n in NUM_FEATS:
            val = row.get(n)
            if val is None:
                row[n] = None
            else:
                try:
                    fv = float(val)
                    if fv.is_integer():
                        row[n] = int(fv)
                    else:
                        row[n] = fv
                except Exception:
                    row[n] = None
        normalized.append(row)
    df = pd.DataFrame(normalized)
    for f in ALL_FEATS:
        if f not in df.columns:
            df[f] = None
    df = df[ALL_FEATS]
    return df


# ------- Endpoints -------
@app.get("/", tags=["health"])
def health():
    return {"service": "Fuel consumption prediction", "status": "OK"}



@app.get("/predict/selection", tags=["predict"])
def predict_selection(
    city08: Optional[float] = Query(None, description="City MPG", ge=1, le=150),
    highway08: Optional[float] = Query(None, description="Highway MPG", ge=1, le=150),
    displ: Optional[float] = Query(None, description="Engine displacement (L)", ge=1, le=10),
    cylinders: Optional[int] = Query(None, description="Number of cylinders", ge=2, le=15),
    
    drive: Optional[DriveType] = Query(None, description="Drive type"),
    fuelType: Optional[FuelType] = Query(None, description="Fuel type"),
    fuelType1: Optional[FuelType1] = Query(None, description="Primary fuel type"),
    fuelType2: Optional[FuelType2] = Query(None, description="Secondary fuel type"),
    model_grouped: Optional[ModelGrouped] = Query(None, description="Vehicle model (grouped)"),
    make_grouped: Optional[MakeGrouped] = Query(None, description="Vehicle manufacturer (grouped)"),
    VClass_grouped: Optional[VClassGrouped] = Query(None, description="Vehicle class (grouped)")
):
    """
    Make prediction with individual parameters and dropdown lists for categorical variables.
    Each categorical parameter shows a dropdown list in the Swagger UI.
    """
    try:
        # Create vehicle data dictionary
        vehicle_data = {
            "city08": city08,
            "highway08": highway08,
            "displ": displ,
            "cylinders": cylinders,
            "drive": drive,
            "fuelType": fuelType,
            "fuelType1": fuelType1,
            "fuelType2": fuelType2,
            "model_grouped": model_grouped,
            "make_grouped": make_grouped,
            "VClass_grouped": VClass_grouped
        }
        
        # Create DataFrame with the single vehicle
        df = pd.DataFrame([vehicle_data])
        
        # Make prediction
        preds = _predict_from_df(df)
        
        return {
            "prediction": preds[0],
            "unit": "mpg",
            "description": "Predicted fuel consumption from selection parameters",
            "status": "success",
            "vehicle_data": vehicle_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


@app.post("/predict/Json", tags=["predict"])
def predict_Json(items: List[FuelInput]):
    rows = pd.DataFrame([i.dict() for i in items])
    try:
        preds = _predict_from_df(rows)
        # Retourner une réponse structurée et riche
        if len(preds) == 1:
            return {
                "prediction": preds[0],
                "unit": "mpg",
                "description": "Predicted fuel consumption",
                "status": "success",
                "input_count": 1
            }
        else:
            return {
                "predictions": preds,
                "unit": "mpg", 
                "description": "Predicted fuel consumption for multiple vehicles",
                "status": "success",
                "input_count": len(preds),
                "average_prediction": sum(preds) / len(preds)
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")




@app.post("/predict/file", tags=["predict"])
async def predict_file(
    file: UploadFile = File(...)
):
    """
    Expects a plain text file (text/plain) containing either:
      - JSON object or JSON list of objects
      - CSV (if it really is CSV)
      - Or plain key=value lines (blocks separated by blank lines) — recommended

    All records in the file will be processed.
    """
    try:
        contents = await file.read()
        # decode as text
        text = contents.decode('utf-8', errors='ignore')
        raw_rows = parse_text_to_rows(text)
        if not raw_rows:
            raise HTTPException(status_code=400, detail="No valid records parsed from the file.")

        df = normalize_rows(raw_rows)
        preds = _predict_from_df(df)

        # Retourner une réponse structurée et riche
        if len(preds) == 1:
            return {
                "prediction": preds[0],
                "unit": "mpg",
                "description": "Predicted fuel consumption from file input",
                "status": "success",
                "file_name": file.filename,
                "input_count": 1
            }
        else:
            return {
                "predictions": preds,
                "unit": "mpg",
                "description": "Predicted fuel consumption for multiple vehicles from file",
                "status": "success", 
                "file_name": file.filename,
                "input_count": len(preds),
                "average_prediction": sum(preds) / len(preds)
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
