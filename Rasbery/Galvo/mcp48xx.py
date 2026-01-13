# SPDX-FileCopyrightText: 2017 Scott Shawcroft, written for Adafruit Industries
# SPDX-FileCopyrightText: Copyright (c) 2024 Steffen Kreutz
#
# SPDX-License-Identifier: MIT
"""
`mcp48xx`
================================================================================

Helper library for the Microchip MCP4801, MCP4811, MCP4821, MCP4802, MCP4812, and MCP4822 digital
to analog converters.


* Author(s): Steffen Kreutz

Implementation Notes
--------------------

**Hardware:**

* `MCP4801 - 8-Bit single channel DAC w/SPI <https://www.microchip.com/en-us/product/mcp4801>`_
* `MCP4811 - 10-Bit single channel DAC w/SPI <https://www.microchip.com/en-us/product/mcp4811>`_
* `MCP4821 - 12-Bit single channel DAC w/SPI <https://www.microchip.com/en-us/product/mcp4821>`_
* `MCP4802 - 8-Bit dual channel DAC w/SPI <https://www.microchip.com/en-us/product/mcp4802>`_
* `MCP4812 - 10-Bit dual channel DAC w/SPI <https://www.microchip.com/en-us/product/mcp4812>`_
* `MCP4822 - 12-Bit dual channel DAC w/SPI <https://www.microchip.com/en-us/product/mcp4822>`_

**Software and Dependencies:**

* Adafruit CircuitPython firmware for the supported boards:
  https://circuitpython.org/downloads
* Adafruit's Bus Device library: https://github.com/adafruit/Adafruit_CircuitPython_BusDevice
"""

__version__ = "1.0.1"
__repo__ = "https://github.com/brushmate/CircuitPython_MCP48XX.git"

# imports

from struct import pack_into
from adafruit_bus_device import spi_device

try:
    from busio import SPI
    from digitalio import DigitalInOut
    from typing import Literal, Optional
except ImportError:
    pass


class _DAC:
    """An internal helper class for writing data to the DAC.

    .. note::
        All instances are created automatically and should not be used by the user.
    """

    def __init__(self, spi_bus: SPI, chip_select: DigitalInOut) -> None:
        self._spi_device = spi_device.SPIDevice(spi_bus, chip_select, baudrate=20000000)

    def write(self, output: bytearray) -> None:
        """Write data to the DAC via the SPI protocol.

        :param bytearray output: the data that will be sent to the DAC.
        """
        with self._spi_device as spi:
            spi.write(output)


class _OutputVoltageLatch:
    """An internal helper class for writing data to the DAC.

    .. note::
        All instances are created automatically and should not be used by the user.
    """

    def __init__(self, latch_input: Optional[DigitalInOut] = None) -> None:
        self._latch_input = latch_input
        if self._latch_input:
            self._latch_input.switch_to_output(True)

    def update(self) -> None:
        """Updates the output voltage(s) of the DAC. On dual channel devices output voltages for
        both channels are updated at the same time.

        .. note::
            This operation has no effect if :attr:`latch_input` is not configured.
        """
        if self._latch_input:
            self._latch_input.value = False
            self._latch_input.value = True


class Channel:
    """An instance of a single channel for a multi-channel DAC.

    :param dac_instance: Instance of the channel object
    :param index: Index of the channel

    .. note::
        All available channels are created automatically
        and should not be created by the user.
    """

    def __init__(
        self,
        index: int,
        resolution: Literal[8, 10, 12],
        dac: _DAC,
    ) -> None:
        self._index = index
        self._resolution = resolution
        self._dac = dac

        self._gain: Literal[0, 1] = 1
        self._shutdown_control: Literal[0, 1] = 1
        self._raw_value = 0

    @property
    def normalized_value(self) -> float:
        """The DAC value as a floating point number in the range 0.0 to 1.0."""
        return self.raw_value / (2**self._resolution - 1)

    @normalized_value.setter
    def normalized_value(self, value: float) -> None:
        if value < 0.0 or value > 1.0:
            raise AttributeError("`normalized_value` must be between 0.0 and 1.0")

        self.raw_value = int(value * (2**self._resolution - 1))

    @property
    def value(self) -> int:
        """The 16-bit scaled current value for the channel. Note that the DAC does not support
        16-bit values so quantization errors will occur.
        """
        return round(self.normalized_value * (2**16 - 1))

    @value.setter
    def value(self, value: int) -> None:
        if value < 0 or value > (2**16 - 1):
            raise AttributeError(
                f"`value` must be a 16-bit integer between 0 and {(2**16 - 1)}"
            )

        # Scale from 16-bit to 12-bit value (quantization errors will occur!).
        self.raw_value = value >> (16 - self._resolution)

    @property
    def raw_value(self) -> int:
        """The native 8-bit, 10-bit, or 12-bit value used by the DAC."""
        return self._raw_value

    @raw_value.setter
    def raw_value(self, value: int) -> None:
        if value < 0 or value > (2**self._resolution - 1):
            res = self._resolution
            raise AttributeError(
                f"`raw_value` must be a {res}-bit integer between 0 and {(2**res - 1)}"
            )
        self._raw_value = value
        self._persist()

    @property
    def gain(self) -> Literal[1, 2]:
        """The gain of the channel.

        With gain set to 1, the output voltage goes from 0v to 2.048V. If a channel's gain is set
        to 2, the voltage goes from 0V to 4.096V. :attr:`gain` must be 1 or 2.
        """
        return 1 if self._gain == 1 else 2

    @gain.setter
    def gain(self, value: Literal[1, 2]) -> None:
        if not value in (1, 2):
            raise AttributeError("`gain` must be 1 or 2")
        self._gain = 1 if value == 1 else 0
        self._persist()

    @property
    def active(self) -> bool:
        """The shutdown state of the channel.

        With active set to True, the output voltage will be available. If it is set to False, the
        output will be connected to a high resistance load and a voltage will not be available.
        """
        return self._shutdown_control == 1

    @active.setter
    def active(self, value: bool) -> None:
        self._shutdown_control = 1 if value else 0
        self._persist()

    def _persist(self) -> None:
        output = self._generate_bytes_with_flags()

        self._dac.write(output)

    def _generate_bytes_with_flags(self) -> bytearray:
        buf = bytearray(2)
        pack_into(">H", buf, 0, self.raw_value << (12 - self._resolution))

        buf[0] |= self._index << 7
        buf[0] |= self._gain << 5
        buf[0] |= self._shutdown_control << 4

        return buf


class _DualChannelDevice:
    def __init__(
        self,
        spi_bus: SPI,
        chip_select: DigitalInOut,
        resolution: Literal[8, 10, 12],
    ) -> None:
        dac = _DAC(spi_bus, chip_select)
        self._channel_a = Channel(0, resolution, dac)
        self._channel_b = Channel(1, resolution, dac)

    @property
    def channel_a(self) -> Channel:
        """Channel A of the DAC. This is a read-only property."""
        return self._channel_a

    @property
    def channel_b(self) -> Channel:
        """Channel B of the DAC. This is a read-only property."""
        return self._channel_b


class MCP4801(Channel, _OutputVoltageLatch):
    """Helper class for the Microchip MCP4801 SPI 8-bit DAC.

    :param ~busio.SPI spi_bus: The SPI bus the MCP4801 is connected to.
    :param digitalio.DigitalInOut chip_select: Board pin the MCP4801 chip select line is connected\
        to.


    **Quickstart: Importing and using the MCP4801**

        Here is an example of using the :class:`MCP4801` class.
        First you will need to import the libraries to use the DAC

        .. code-block:: python

            import board
            import busio
            import digitalio
            import mcp48xx

        Once this is done you can define your `board.SPI` object and define your sensor object

        .. code-block:: python

            spi = busio.SPI(board.SCK, board.MOSI)   # The MCP4801 has no MISO pin
            cs = digitalio.DigitalInOut(board.IO1)
            mcp4801 = mcp48xx.MCP4801(spi, cs)

        Now you can give values to the DAC

        .. code-block:: python

            mcp4801.value = 65535  # Voltage = 2.048 (if gain=1)
    """

    def __init__(
        self,
        spi_bus: SPI,
        chip_select: DigitalInOut,
        latch_input: Optional[DigitalInOut] = None,
    ) -> None:
        Channel.__init__(self, 0, 8, _DAC(spi_bus, chip_select))
        _OutputVoltageLatch.__init__(self, latch_input)


class MCP4811(Channel, _OutputVoltageLatch):
    """Helper class for the Microchip MCP4811 SPI 10-bit DAC.

    :param ~busio.SPI spi_bus: The SPI bus the MCP4811 is connected to.
    :param digitalio.DigitalInOut chip_select: Board pin the MCP4811 chip select line is connected
        to.


    **Quickstart: Importing and using the MCP4811**

        Here is an example of using the :class:`MCP4811` class.
        First you will need to import the libraries to use the DAC

        .. code-block:: python

            import board
            import busio
            import digitalio
            import mcp48xx

        Once this is done you can define your `board.SPI` object and define your sensor object

        .. code-block:: python

            spi = busio.SPI(board.SCK, board.MOSI)   # The MCP4811 has no MISO pin
            cs = digitalio.DigitalInOut(board.IO1)
            mcp4811 = mcp48xx.MCP4811(spi, cs)

        Now you can give values to the DAC

        .. code-block:: python

            mcp4811.value = 65535  # Voltage = 2.048 (if gain=1)
    """

    def __init__(
        self,
        spi_bus: SPI,
        chip_select: DigitalInOut,
        latch_input: Optional[DigitalInOut] = None,
    ) -> None:
        Channel.__init__(self, 0, 10, _DAC(spi_bus, chip_select))
        _OutputVoltageLatch.__init__(self, latch_input)


class MCP4821(Channel, _OutputVoltageLatch):
    """Helper class for the Microchip MCP4821 SPI 12-bit DAC.

    :param ~busio.SPI spi_bus: The SPI bus the MCP4821 is connected to.
    :param digitalio.DigitalInOut chip_select: Board pin the MCP4821 chip select line is connected
        to.


    **Quickstart: Importing and using the MCP4821**

        Here is an example of using the :class:`MCP4821` class.
        First you will need to import the libraries to use the DAC

        .. code-block:: python

            import board
            import busio
            import digitalio
            import mcp48xx

        Once this is done you can define your `board.SPI` object and define your sensor object

        .. code-block:: python

            spi = busio.SPI(board.SCK, board.MOSI)   # The MCP4821 has no MISO pin
            cs = digitalio.DigitalInOut(board.IO1)
            mcp4821 = mcp48xx.MCP4821(spi, cs)

        Now you can give values to the DAC

        .. code-block:: python

            mcp4821.value = 65535  # Voltage = 2.048 (if gain=1)
    """

    def __init__(
        self,
        spi_bus: SPI,
        chip_select: DigitalInOut,
        latch_input: Optional[DigitalInOut] = None,
    ) -> None:
        Channel.__init__(self, 0, 12, _DAC(spi_bus, chip_select))
        _OutputVoltageLatch.__init__(self, latch_input)


class MCP4802(_DualChannelDevice, _OutputVoltageLatch):
    """Helper class for the Microchip MCP4802 SPI 8-bit Dual DAC.

    :param ~busio.SPI spi_bus: The SPI bus the MCP4802 is connected to.
    :param digitalio.DigitalInOut chip_select: Board pin the MCP4802 chip select line is connected
        to.


    **Quickstart: Importing and using the MCP4802**

        Here is an example of using the :class:`MCP4802` class.
        First you will need to import the libraries to use the DAC

        .. code-block:: python

            import board
            import busio
            import digitalio
            import mcp48xx

        Once this is done you can define your `board.SPI` object and define your sensor object

        .. code-block:: python

            spi = busio.SPI(board.SCK, board.MOSI)   # The MCP4802 has no MISO pin
            cs = digitalio.DigitalInOut(board.IO1)
            mcp4802 = mcp48xx.MCP4802(spi, cs)

        Now you can give values to the different channels

        .. code-block:: python

            mcp4802.channel_a.value = 65535  # Voltage = 2.048 (if gain=1)
            mcp4802.channel_b.value = int(65535 / 2)  # Voltage = 1.024 (if gain=1)
    """

    def __init__(
        self,
        spi_bus: SPI,
        chip_select: DigitalInOut,
        latch_input: Optional[DigitalInOut] = None,
    ) -> None:
        _DualChannelDevice.__init__(self, spi_bus, chip_select, 8)
        _OutputVoltageLatch.__init__(self, latch_input)


class MCP4812(_DualChannelDevice, _OutputVoltageLatch):
    """Helper class for the Microchip MCP4812 SPI 10-bit Dual DAC.

    :param ~busio.SPI spi_bus: The SPI bus the MCP4812 is connected to.
    :param digitalio.DigitalInOut chip_select: Board pin the MCP4812 chip select line is connected
        to.


    **Quickstart: Importing and using the MCP4812**

        Here is an example of using the :class:`MCP4812` class.
        First you will need to import the libraries to use the DAC

        .. code-block:: python

            import board
            import busio
            import digitalio
            import mcp48xx

        Once this is done you can define your `board.SPI` object and define your sensor object

        .. code-block:: python

            spi = busio.SPI(board.SCK, board.MOSI)   # The MCP4812 has no MISO pin
            cs = digitalio.DigitalInOut(board.IO1)
            mcp4812 = mcp48xx.MCP4812(spi, cs)

        .. code-block:: python

            mcp4812.channel_a.value = 65535  # Voltage = 2.048 (if gain=1)
            mcp4812.channel_b.value = int(65535 / 2)  # Voltage = 1.024 (if gain=1)
    """

    def __init__(
        self,
        spi_bus: SPI,
        chip_select: DigitalInOut,
        latch_input: Optional[DigitalInOut] = None,
    ) -> None:
        _DualChannelDevice.__init__(self, spi_bus, chip_select, 8)
        _OutputVoltageLatch.__init__(self, latch_input)


class MCP4822(_DualChannelDevice, _OutputVoltageLatch):
    """Helper class for the Microchip MCP4822 SPI 12-bit Dual DAC.

    :param ~busio.SPI spi_bus: The SPI bus the MCP4822 is connected to.
    :param digitalio.DigitalInOut chip_select: Board pin the MCP4822 chip select line is connected
        to.


    **Quickstart: Importing and using the MCP4822**

        Here is an example of using the :class:`MCP4822` class.
        First you will need to import the libraries to use the DAC

        .. code-block:: python

            import board
            import busio
            import digitalio
            import mcp48xx

        Once this is done you can define your `board.SPI` object and define your sensor object

        .. code-block:: python

            spi = busio.SPI(board.SCK, board.MOSI)   # The MCP4822 has no MISO pin
            cs = digitalio.DigitalInOut(board.IO1)
            mcp4822 = mcp48xx.MCP4822(spi, cs)

        Now you can give values to the different channels

        .. code-block:: python

            mcp4822.channel_a.value = 65535  # Voltage = 2.048 (if gain=1)
            mcp4822.channel_b.value = int(65535 / 2)  # Voltage = 1.024 (if gain=1)
    """

    def __init__(
        self,
        spi_bus: SPI,
        chip_select: DigitalInOut,
        latch_input: Optional[DigitalInOut] = None,
    ) -> None:
        _DualChannelDevice.__init__(self, spi_bus, chip_select, 12)
        _OutputVoltageLatch.__init__(self, latch_input)