from dataclasses import dataclass, asdict

# Define a schema for a detection results
@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: list[int]  # [x_min, y_min, width, height]

    def to_dict(self):        
      return asdict(self)
    
# Define a schema for a detection message
@dataclass
class DetectionMessage:
   frame_id: int
   detections: list[Detection]

   def to_dict(self):
       return asdict(self)
   
@dataclass
class Paquet:
   id: int
   