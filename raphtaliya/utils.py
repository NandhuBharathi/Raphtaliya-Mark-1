from datetime import datetime

def separator(length=60):
    print("=" * length)

def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def show_upgrade(module, version):
    separator()
    print(f"[{timestamp()}]")
    print("Raphtaliya Mark-1")
    print(f"Module  : {module}")
    print(f"Version : {version}")
    print("Status  : Upgrade Successfully")
    separator()
