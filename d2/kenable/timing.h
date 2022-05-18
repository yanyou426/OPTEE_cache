/* See LICENSE file for license and copyright information */

#ifndef ARM_V7_TIMING_H
#define ARM_V7_TIMING_H
#endif 

#include "configuration.h"

#define ARMV7_PMCR_E       (1 << 0) /* Enable all counters */
#define ARMV7_PMCR_P       (1 << 1) /* Reset all counters */
#define ARMV7_PMCR_C       (1 << 2) /* Cycle counter reset */
#define ARMV7_PMCR_D       (1 << 3) /* Cycle counts every 64th cpu cycle */
#define ARMV7_PMCR_X       (1 << 4) /* Export to ETM */

#define ARMV7_PMCNTENSET_C (1 << 31) /* Enable cycle counter */

#define ARMV7_PMOVSR_C     (1 << 31) /* Overflow bit */