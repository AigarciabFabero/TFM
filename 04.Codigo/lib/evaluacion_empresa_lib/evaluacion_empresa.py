def hola_mundo():
    print("¡Hola Mundo!")

def saludar(nombre):
    print(f"¡Hola {nombre}!")
    
def despedir(nombre=""):
    if nombre:
        print(f"¡Adiós {nombre}!")
    else:
        print("¡Adiós!")
