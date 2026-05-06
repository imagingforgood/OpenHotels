from .registry import MODEL_REGISTRY, get_model

# Import model modules to trigger @register_model decorators.
# Add new models here as they are implemented.
from . import epshn   
from . import clip    
from . import siglip  
from . import dino    
from . import vit    
