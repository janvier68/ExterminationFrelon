import json
import board

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def gpio_from_string(pin: str):
    """
    Convertit "D17" -> board.D17
    """
    if not isinstance(pin, str):
        raise TypeError("laser_pin doit être une string du type 'D17'")
    try:
        return getattr(board, pin)
    except AttributeError as e:
        raise ValueError(f"Pin board.{pin} introuvable") from e
