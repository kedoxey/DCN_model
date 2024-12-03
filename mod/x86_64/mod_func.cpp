#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;
#if defined(__cplusplus)
extern "C" {
#endif

extern void _izhi2003a_reg(void);
extern void _izhi2003b_reg(void);
extern void _izhi2007a_reg(void);
extern void _izhi2007b_reg(void);
extern void _spikeout_reg(void);
extern void _vecevent_reg(void);
extern void _xspikeout_reg(void);

void modl_reg() {
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");
    fprintf(stderr, " \"./izhi2003a.mod\"");
    fprintf(stderr, " \"./izhi2003b.mod\"");
    fprintf(stderr, " \"./izhi2007a.mod\"");
    fprintf(stderr, " \"./izhi2007b.mod\"");
    fprintf(stderr, " \"./spikeout.mod\"");
    fprintf(stderr, " \"./vecevent.mod\"");
    fprintf(stderr, " \"./xspikeout.mod\"");
    fprintf(stderr, "\n");
  }
  _izhi2003a_reg();
  _izhi2003b_reg();
  _izhi2007a_reg();
  _izhi2007b_reg();
  _spikeout_reg();
  _vecevent_reg();
  _xspikeout_reg();
}

#if defined(__cplusplus)
}
#endif
