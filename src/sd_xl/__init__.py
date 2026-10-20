import torch
from diffusers import AutoPipelineForText2Image

MODELO = "stabilityai/stable-diffusion-xl-base-1.0"
SALIDA = "imagen.png"
NEGATIVE_PROMPT = (
    "blurry, low resolution, low quality, jpeg artifacts, bad anatomy, "
    "deformed hands, extra fingers, missing limbs, poorly drawn face, "
    "watermark, signature, text, username, error"
)


def elegir_dispositivo():
    # cuda - nvidia | mps - Mac con chip M | cpu - cualquier equipo
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    dispositivo = elegir_dispositivo()
    # float16 en GPU; en CPU bfloat16 para que SDXL quepa en ~8 GB de RAM
    # (en float32 necesita mas de 14 GB)
    tipo = torch.float16 if dispositivo == "cuda" else torch.bfloat16

    print(f"Cargando modelo en {dispositivo} ...")
    modelo = AutoPipelineForText2Image.from_pretrained(MODELO, torch_dtype=tipo)
    modelo = modelo.to(dispositivo)

    prompt = input("Escribe el prompt de la imagen: ")

    print("Generando la imagen...")
    imagen = modelo(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=30,  # Pasos de refinamiento de la imagen
        guidance_scale=7.0,  # Que tan fiel sera al prompt
        height=1024,  # resolucion nativa de SDXL
        width=1024,
    ).images[0]

    imagen.save(SALIDA)
    print(f"Imagen guardada en {SALIDA}")


if __name__ == "__main__":
    main()
