from random import randint
from typing import Iterator
from unittest import TestCase

from lcls_tools.common.controls.pyepics.utils import make_mock_pv
from lcls_tools.superconducting.sc_linac import CavityIterator
from lcls_tools.superconducting.sc_linac_utils import (
    HW_MODE_OFFLINE_VALUE,
    HW_MODE_ONLINE_VALUE,
)
from sel_phase_linac import SELCavity

cavity_iterator: Iterator[SELCavity] = CavityIterator(
    cavity_class=SELCavity
).non_hl_iterator


class TestSELCavity(TestCase):
    def test_sel_phase_offset(self):
        cavity = next(cavity_iterator)
        offset = randint(-10, 10)
        cavity._sel_poff_pv_obj = make_mock_pv(get_val=offset)
        self.assertEqual(cavity.sel_phase_offset, offset)

    def test_i_waveform(self):
        cavity = next(cavity_iterator)
        wf = [randint(-10, 10) for i in range(10)]
        cavity._i_waveform_pv_obj = make_mock_pv(get_val=wf)

        self.assertEqual(cavity.i_waveform, wf)

    def test_q_waveform(self):
        cavity = next(cavity_iterator)
        wf = [randint(-10, 10) for i in range(10)]
        cavity._q_waveform_pv_obj = make_mock_pv(get_val=wf)

        self.assertEqual(cavity.q_waveform, wf)

    def test_straighten_cheeto_offline(self):
        cavity = next(cavity_iterator)
        cavity._hw_mode_pv_obj = make_mock_pv(get_val=HW_MODE_OFFLINE_VALUE)
        self.assertFalse(cavity.straighten_cheeto())

    def test_straighten_cheeto_low_amp(self):
        cavity = next(cavity_iterator)
        cavity._hw_mode_pv_obj = make_mock_pv(get_val=HW_MODE_ONLINE_VALUE)
        cavity._aact_pv_obj = make_mock_pv(get_val=0.5)
        self.assertFalse(cavity.straighten_cheeto())
