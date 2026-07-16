#!/usr/bin/env python3
"""
Script de entrada para SpeedrEye Modular
Ejecuta el sistema con las importaciones correctas
"""

import sys
import os
from pathlib import Path

# Añadir el directorio actual al path para poder importar speedreye_modular
sys.path.insert(0, str(Path(__file__).parent))

# Importar y ejecutar el main del módulo
from speedreye_modular.main import main

if __name__ == "__main__":
    main()