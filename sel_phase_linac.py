import logging
import os
import time
from typing import Optional

import numpy as np
from scipy import stats

from lcls_tools.common.controls.pyepics.utils import PV
from lcls_tools.common.logger.logger import custom_logger, FORMAT_STRING
from lcls_tools.superconducting.sc_linac import (
    Cavity,
    Machine,
)

MAX_STEP = 5
MULT = -51.0471


class SELCavity(Cavity):
    def __init__(
        self,
        cavity_num,
        rack_object,
    ):
        super().__init__(cavity_num=cavity_num, rack_object=rack_object)
        self.q_waveform_pv: str = self.pv_addr("CTRL:QWF")
        self._q_waveform_pv_obj: Optional[PV] = None

        self.i_waveform_pv: str = self.pv_addr("CTRL:IWF")
        self._i_waveform_pv_obj: Optional[PV] = None

        self.sel_poff_pv: str = self.pv_addr("SEL_POFF")
        self._sel_poff_pv_obj: Optional[PV] = None

        self.logger = custom_logger(f"{self} SEL Phase Opt Logger")
        self.logfile = (
            f"logfiles/cm{self.cryomodule.name}/{self.number}_sel_phase_opt.log"
        )
        os.makedirs(os.path.dirname(self.logfile), exist_ok=True)

        self.file_handler = logging.FileHandler(self.logfile, mode="a")
        self.file_handler.setFormatter(logging.Formatter(FORMAT_STRING))
        self.logger.addHandler(self.file_handler)

    @property
    def sel_poff_pv_obj(self) -> PV:
        if not self._sel_poff_pv_obj:
            self._sel_poff_pv_obj = PV(self.sel_poff_pv)
        return self._sel_poff_pv_obj

    @property
    def sel_phase_offset(self):
        return self.sel_poff_pv_obj.get()

    @property
    def i_waveform(self):
        if not self._i_waveform_pv_obj:
            self._i_waveform_pv_obj = PV(self.i_waveform_pv)
        return self._i_waveform_pv_obj.get()

    @property
    def q_waveform(self):
        if not self._q_waveform_pv_obj:
            self._q_waveform_pv_obj = PV(self.q_waveform_pv)
        return self._q_waveform_pv_obj.get()

    def straighten_cheeto(self) -> bool:
        """
        :return: True if wanted to take a step larger than MAX_STEP
        """

        if not self.is_online or self.aact <= 1:
            return False

        startVal = self.sel_phase_offset
        iwf = self.i_waveform
        qwf = self.q_waveform
        large_step = False

        [slop, inter] = stats.siegelslopes(iwf, qwf)

        if not np.isnan(slop):
            chisum = 0
            for nn, yy in enumerate(iwf):
                chisum += (yy - (slop * iwf[nn] + inter)) ** 2 / (
                    slop * iwf[nn] + inter
                )

            step = slop * MULT
            if abs(step) > MAX_STEP:
                step = MAX_STEP * np.sign(step)
                prefix = "\033[91m"
                suffix = "\033[0m"
                large_step = True
                self.logger.warning(f"{prefix}Large step taken{suffix}")
            else:
                prefix = ""
                suffix = ""
            if startVal + step < -180:
                step = step + 360
            elif startVal + step > 180:
                step = step - 360

            timi = time.localtime()
            current_time = time.strftime("%m/%d %H:%M ", timi)
            self.logger.info(
                f"{prefix}{current_time}{self}{suffix}  step: {step:5.2f} chi^2: {chisum:.2g}"
            )

            self.sel_poff_pv_obj.put(startVal + step)
            return large_step

        else:
            self.logger.warning(f"{self} slope is NaN, skipping")


SEL_MACHINE: Machine = Machine(cavity_class=SELCavity)
