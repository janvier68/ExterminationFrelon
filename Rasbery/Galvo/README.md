## GalvoController (DAC MCP4822 + Laser)

### Objectif

Piloter un galvo 2 axes via un DAC **MCP4822 (SPI)** et contrôler un laser (GPIO) :

* Conversion angle → valeur DAC
* Commande X/Y en continu
* Gestion ON/OFF laser
* Arrêt sécurisé

## Classe `GalvoController`

```python
GalvoController(
    max_angle_deg=40.0,
    laser_pin=board.D17,
    gain=1,
    safe_start=True,
    max_code=4095
)
```

### Paramètres

* `max_angle_deg` : angle max autorisé (limitation automatique)
* `laser_pin` : pin GPIO du laser
* `gain` : gain du DAC (MCP4822)
* `safe_start` : met X/Y à 0 au démarrage
* `max_code` : résolution DAC (12 bits = 4095)

## Fonctions

### `angle_to_dac(angle_deg, max_angle_deg=20.0)`

Convertit un angle (de `-max` à `+max`) en code DAC `[0..max_code]` avec clamp.

### `set_angles(theta_x, theta_y)`

Envoie les angles X/Y au DAC :

* channel A = X
* channel B = Y

### `laser_on()`

Active le laser (GPIO = HIGH).

### `laser_off()`

Coupe le laser (GPIO = LOW) et recentre le galvo (X=0, Y=0).

### `shutdown()`

Arrêt sécurisé :

* laser OFF
* recentrage X/Y
* petite pause (`sleep(0.1)`)
