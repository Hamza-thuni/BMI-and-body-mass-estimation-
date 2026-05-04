import time
import random
from config import Config

class MockBMIPipeline:
    """
    Simulates the V9 Hybrid Inference Pipeline.
    This mock class instantly returns realistic data so we can develop the UI
    without the overhead of loading heavy MediaPipe and PyTorch models.
    """
    
    def __init__(self, delay=Config.MOCK_DELAY):
        self.delay = delay
        print(f"[MockBMIPipeline] Initialized with delay={self.delay}s")
        
    def predict(self, age, sex_is_male):
        """
        Simulates inference time and returns mock body composition data.
        """
        # 1. Simulate the V9 pipeline 'thinking' state (extracting features, running XGBoost/Ridge)
        time.sleep(self.delay)
        
        # 2. Generate a plausible randomized baseline
        # Auto-detect height around 1.7m (170cm)
        height_m = round(random.uniform(1.55, 1.95), 2)
        height_cm = int(height_m * 100)
        
        # Generate a BMI between 18.0 and 32.0
        bmi = round(random.uniform(18.0, 32.0), 1)
        
        # Calculate Weight: BMI = Weight / Height^2 => Weight = BMI * Height^2
        weight_kg = round(bmi * (height_m ** 2), 1)
        
        # 3. Calculate Body Fat % using the Deurenberg formula
        # BFP = (1.20 × BMI) + (0.23 × Age) - (10.8 × Sex) - 5.4 
        # where Sex = 1 for male, 0 for female
        sex_val = 1 if sex_is_male else 0
        body_fat_pct = (1.20 * bmi) + (0.23 * age) - (10.8 * sex_val) - 5.4
        
        # Clip body fat percentage to reasonable bounds
        body_fat_pct = max(3.0, min(50.0, body_fat_pct))
        body_fat_pct = round(body_fat_pct, 1)
        
        # 4. Calculate Fat Mass vs Lean Mass
        fat_mass = round(weight_kg * (body_fat_pct / 100.0), 1)
        lean_mass = round(weight_kg - fat_mass, 1)
        
        # 5. Determine Health Category
        if body_fat_pct < (10.0 if sex_is_male else 20.0):
            category = "Lean"
        elif body_fat_pct < (20.0 if sex_is_male else 30.0):
            category = "Healthy"
        elif body_fat_pct < (25.0 if sex_is_male else 35.0):
            category = "Overweight"
        else:
            category = "Obese"
            
        return {
            "height_m": height_m,
            "weight_kg": weight_kg,
            "bmi": bmi,
            "body_fat_pct": body_fat_pct,
            "fat_mass": fat_mass,
            "lean_mass": lean_mass,
            "category": category
        }
