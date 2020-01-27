# DynBinEv

**DYNamical BInary EVolution**

**Language:** python script for binary evolution with N-body dynamics

**version:** 1.e-6

A collaborative effort to construct a binary population synthesis code
in which the orbital evolution is calculated using a general-purpose
N-body code and the stellar evolution by a general purpose stellar
evolution code.

The objective is to eventually, be able to resplace the binary
population synthesis modeules in N-body codes by more general-purpose
(but slower) prescription for the binary evolution in which the
dynamics is taken care of by the N-body code.

Initial discussions started on 27 January 2020 between:
Nicola Giacobbo,
Iorio Giuliano,
Simon Portegies Zwart,
Steven Rieder,
Alessandro Trani,
and
Long Wang

**Current content are 3 scripts:**

-dynbin_simple.py:
       Simplest form of integrating a binary without any mass loss.
-dynbin_massloss.py:
       Simplest for of binary evolution in which the binary orbit is
       integrated with a 4th-order Hermite scheme. Stellar mass loss
       (currently constant with time) is directly incorporated into
       the N-body code via a channel.
-dynbin_massloss_bridge.py:
       Binary evolution in which the binary orbit is
       integrated with a 4th-order Hermite scheme and the stellar mass loss
       (currently constant with time) is incorporated using the drift-operator
       in the classic bridge. 

**Licence: MIT**