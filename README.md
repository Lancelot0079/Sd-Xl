# SD-XL: generación de imágenes con Stable Diffusion XL

Genera una imagen a partir de un texto (*prompt*) usando [`stabilityai/stable-diffusion-xl-base-1.0`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) con `diffusers` y PyTorch. Funciona en CPU, y usa GPU NVIDIA (CUDA) o Mac (MPS) si están disponibles.

## Requisitos

- [uv](https://docs.astral.sh/uv/) y Python 3.12 o superior
- Conexión a internet la primera vez (descarga el modelo, más de 6 GB)
- Al menos 8 GB de RAM libres en CPU (16 GB recomendados)

## Ejecución

```bash
uv sync
uv run sd-xl
```

El programa pide el prompt (conviene en inglés) y guarda el resultado en `imagen.png`. En CPU puede tardar varios minutos por imagen; con una GPU NVIDIA, segundos.

## Parámetros usados y por qué

| Parámetro | Valor | Motivo |
|---|---|---|
| `height` / `width` | 1024 | Resolución nativa de SDXL |
| `num_inference_steps` | 30 | Equilibrio entre calidad y tiempo |
| `guidance_scale` | 7.0 | Fidelidad al prompt (SDXL usa 5 a 8) |
| `negative_prompt` | lista fija | Evita desenfoque, manos deformadas, marcas de agua, etc. |
| `torch_dtype` | float16 en GPU, bfloat16 en CPU | float32 necesita más de 14 GB de RAM; bfloat16 la reduce a la mitad |

## Usar la GPU NVIDIA

Por defecto `uv` instala PyTorch para CPU en Windows. Para usar CUDA hay que instalar torch con CUDA siguiendo la [guía de uv para PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/). El código detecta la GPU automáticamente.

## Estructura

```
├── pyproject.toml
├── uv.lock
├── src/sd_xl/__init__.py   # función main()
└── imagen.png              # imagen de ejemplo
```
