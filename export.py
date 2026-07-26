#!/usr/bin/env python3
"""
Exporta la carpeta src/ (con contenido de todos los archivos) y solo los nombres de los archivos en models/.
Incluye carpetas ocultas (como __pycache__ si existen, aunque normalmente se ignoran).
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import fnmatch

# ============================================
# CONFIGURACIÓN
# ============================================

PROJECT_ROOT = Path.cwd()
OUTPUT_FILE = "src_export.txt"

# Carpetas a IGNORAR en src/ (solo lo esencial)
IGNORE_DIRS = {
    '__pycache__',
    '.pytest_cache', 
    '.mypy_cache', 
    '.ipynb_checkpoints',
    'node_modules',
}

# Patrones de archivos a IGNORAR en src/
IGNORE_PATTERNS = [
    '*.pyc', '*.pyo', '*.pyd',
    '*.so', '*.dll', '*.exe',
    '*.bin', '*.zip', '*.tar.gz',
    '*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff',
    '*.mp4', '*.avi', '*.mov', '*.mkv',
    '*.pth', '*.pt', '*.onnx', '*.engine',
    '*.ply', '*.pkl', '*.npy', '*.npz',
    '*.db', '*.sqlite', '*.sqlite3',
    '*.log', '*.tmp', '*.temp',
]

# Extensiones a INCLUIR en src/
INCLUDE_EXTENSIONS = {
    '.py', '.txt', '.md', '.json', '.yaml', '.yml', '.xml',
    '.html', '.css', '.js', '.sh', '.bat', '.cfg', '.conf',
    '.ini', '.toml', '.ipynb', '.csv', '.gitignore',
    '.python-version', '.env', '.gitmodules', '.gitattributes',
    '.in', '.am', '.cmake', '.txt', '.rst',
}

# ============================================
# FUNCIONES
# ============================================

def should_ignore_src(path):
    """Determina si un path en src/ debe ser ignorado."""
    name = path.name
    
    # Ignorar directorios
    if path.is_dir():
        for ignore in IGNORE_DIRS:
            if fnmatch.fnmatch(name, ignore):
                return True
        return False
    
    # Ignorar por patrón
    for pattern in IGNORE_PATTERNS:
        if fnmatch.fnmatch(name, pattern):
            return True
    
    # Si no tiene extensión, incluir
    if not path.suffix:
        return False
    
    # Incluir solo extensiones de texto/código
    if path.suffix not in INCLUDE_EXTENSIONS:
        return True
    
    # Ignorar archivos grandes (>1MB)
    try:
        if path.stat().st_size > 1 * 1024 * 1024:
            return True
    except:
        pass
    
    return False


def get_file_tree(directory, prefix="", is_last=True, ignore_func=None):
    """Genera un árbol de archivos completo."""
    lines = []
    
    if ignore_func is None:
        ignore_func = should_ignore_src
    
    try:
        items = sorted([p for p in directory.iterdir() if not ignore_func(p)])
    except PermissionError:
        return lines
    
    for i, path in enumerate(items):
        is_last_item = (i == len(items) - 1)
        connector = "└── " if is_last_item else "├── "
        
        if path.is_dir():
            lines.append(f"{prefix}{connector}{path.name}/")
            extension = "    " if is_last_item else "│   "
            lines.extend(get_file_tree(path, prefix + extension, is_last_item, ignore_func))
        else:
            lines.append(f"{prefix}{connector}{path.name}")
    
    return lines


def export_src_only(output_file):
    """Exporta src/ con contenido y models/ solo nombres."""
    
    src_dir = PROJECT_ROOT / "src"
    models_dir = PROJECT_ROOT / "models"
    
    print(f"📂 Exportando desde: {PROJECT_ROOT}")
    print(f"📄 Archivo de salida: {output_file}")
    print("")
    print("📌 src/: contenido completo de todos los archivos")
    print("📌 models/: solo nombres de archivos (sin contenido)")
    print("")
    
    if not src_dir.exists():
        print(f"❌ Error: No se encuentra src/ en {PROJECT_ROOT}")
        return
    
    total_files = 0
    total_size = 0
    
    with open(output_file, 'w', encoding='utf-8') as out:
        # Cabecera
        out.write("=" * 80 + "\n")
        out.write("EXPORTACIÓN DE src/ + listado de models/\n")
        out.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        out.write(f"Proyecto: {PROJECT_ROOT.absolute()}\n")
        out.write("=" * 80 + "\n\n")
        
        # ============================================
        # PARTE 1: ESTRUCTURA DE src/
        # ============================================
        out.write("📁 ESTRUCTURA DE src/\n")
        out.write("-" * 80 + "\n")
        out.write("src/\n")
        tree_lines = get_file_tree(src_dir, "    ", ignore_func=should_ignore_src)
        for line in tree_lines:
            out.write(line + "\n")
        out.write("\n" + "=" * 80 + "\n\n")
        
        # ============================================
        # PARTE 2: CONTENIDO DE ARCHIVOS EN src/
        # ============================================
        out.write("📄 CONTENIDO DE ARCHIVOS EN src/\n")
        out.write("-" * 80 + "\n\n")
        
        for root, dirs, files in os.walk(src_dir):
            # Filtrar directorios ignorados
            dirs[:] = [d for d in dirs if not should_ignore_src(Path(root) / d)]
            
            for file in sorted(files):
                file_path = Path(root) / file
                
                # Verificar si debe ser ignorado
                if should_ignore_src(file_path):
                    continue
                
                # Mostrar ruta relativa
                rel_path = file_path.relative_to(PROJECT_ROOT)
                
                # Escribir cabecera del archivo
                out.write(f"\n{'='*80}\n")
                out.write(f"📄 {rel_path}\n")
                out.write(f"📁 Ruta: {file_path.absolute()}\n")
                out.write(f"📄 Extensión: {file_path.suffix or 'sin extensión'}\n")
                out.write(f"📏 Tamaño: {file_path.stat().st_size} bytes\n")
                out.write(f"{'='*80}\n\n")
                
                # Leer y escribir contenido
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        out.write(content)
                        if not content.endswith('\n'):
                            out.write('\n')
                except UnicodeDecodeError:
                    out.write("⚠️ Error de codificación (archivo binario o UTF-8 inválido)\n")
                except Exception as e:
                    out.write(f"❌ Error al leer el archivo: {e}\n")
                
                total_files += 1
                total_size += file_path.stat().st_size
        
        # ============================================
        # PARTE 3: LISTADO DE models/ (SOLO NOMBRES)
        # ============================================
        out.write("\n" + "=" * 80 + "\n")
        out.write("📁 LISTADO DE models/ (SOLO NOMBRES DE ARCHIVOS)\n")
        out.write("-" * 80 + "\n")
        
        if models_dir.exists():
            # Listar todos los archivos en models/ recursivamente
            for root, dirs, files in os.walk(models_dir):
                # Ignorar __pycache__ y similares
                dirs[:] = [d for d in dirs if d not in ['__pycache__']]
                
                rel_root = Path(root).relative_to(PROJECT_ROOT)
                out.write(f"\n📂 {rel_root}/\n")
                for file in sorted(files):
                    if not file.startswith('.'):
                        # Mostrar tamaño del archivo
                        try:
                            size = Path(root, file).stat().st_size
                            size_str = f"({size / 1024:.1f} KB)" if size > 1024 else f"({size} B)"
                        except:
                            size_str = ""
                        out.write(f"   ├── {file} {size_str}\n")
        else:
            out.write("⚠️ La carpeta models/ no existe.\n")
        
        # ============================================
        # RESUMEN FINAL
        # ============================================
        out.write("\n" + "=" * 80 + "\n")
        out.write("📊 RESUMEN DE EXPORTACIÓN\n")
        out.write("-" * 80 + "\n")
        out.write(f"Total de archivos exportados (src/): {total_files}\n")
        out.write(f"Tamaño total: {total_size / 1024 / 1024:.2f} MB\n")
        out.write("=" * 80 + "\n")
    
    print(f"✅ Exportación completada: {output_file}")
    print(f"📊 Archivos exportados (src/): {total_files}")
    print(f"📊 Tamaño total: {total_size / 1024 / 1024:.2f} MB")
    print("")
    print("💡 Ahora puedes pasar el contenido de este archivo para revisión.")
    print("   - src/: ✅ contenido completo")
    print("   - models/: ✅ solo nombres de archivos")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Exporta src/ con contenido y models/ solo nombres."
    )
    parser.add_argument("--output", "-o", type=str, default="src_export.txt",
                        help="Archivo de salida")
    parser.add_argument("--dir", "-d", type=str, default=None,
                        help="Directorio raíz (por defecto: actual)")
    
    args = parser.parse_args()
    
    global PROJECT_ROOT
    
    if args.dir:
        PROJECT_ROOT = Path(args.dir)
        if not PROJECT_ROOT.exists():
            print(f"❌ Error: El directorio '{args.dir}' no existe.")
            return
    
    export_src_only(args.output)


if __name__ == "__main__":
    main()