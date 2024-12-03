NEURON {
:  POINT_PROCESS SpikeOut
  POINT_PROCESS XSpikeOut
:  GLOBAL thresh, refrac, vrefrac, grefrac
  GLOBAL thresh, refrac, p, vrefrac0, vrefrac1, grefrac
  NONSPECIFIC_CURRENT i
  RANGE g, vrev
}

PARAMETER {
  thresh = -55 (millivolt)
  refrac = 3 (ms)
:  vrefrac = -60 (millivolt)
  vrefrac0 = -60 (millivolt)
  vrefrac1 = 10 (millivolt)
  p = 0 : fraction of refractory interval 
        : during which vrev == vrefrac1
        : if p = 0, vrev = vrefrac0 throughout the entire refractory interval
        : if p = 1, vrev = vrefrac1 throughout the entire refractory interval
  grefrac = 100 (microsiemens) :clamp to vrefrac
}

ASSIGNED {
  i (nanoamp)
  v (millivolt)
  vrev (millivolt)
  g (microsiemens)
}

INITIAL {
:  net_send(0, 3) : because there were three actions that had to occur--
    : initialization, starting a spike, and ending a spike.
    : These were controlled by self events with a flag variable 
    : that could have one of three values--
    : 1 meant that v crossed spike threshold, so it is time to enter the "spike sequence"
    : 2 meant it is time to end the spike sequence
    : 3 meant "execute the WATCH statement"
    : Now there are four actions--
    : initialization, starting a spike, changing vrev, and ending a spike.
  if (refrac<0) { : force refrac >=0
    refrac = 0
  }
  if (grefrac<0) { : force grefrac >=0
    grefrac = 0
  }
  if (p<0) { : force 0<=p<=1
    p = 0
  }
  if (p>1) {
    p = 1
  }
  g = 0
  vrev = v
: printf("INITIAL block:  t %f  vrev %f  g %f\n", t, vrev, g)
  net_send(0, 4)
}

BREAKPOINT {
:  i = g*(v - vrefrac)
  i = g*(v - vrev)
}

COMMENT
NET_RECEIVE(w) {
  if (flag == 1) {
    net_event(t)
    net_send(refrac, 2)
    v = vrefrac
    g = grefrac
  }else if (flag == 2) {
    g = 0
  }else if (flag == 3) {
    WATCH (v > thresh) 1
  }  
}
ENDCOMMENT

NET_RECEIVE(w) {
  if (flag == 1) {
    net_event(t)
    net_send(refrac*(1-p), 2)
:    v = vrefrac
    vrev = vrefrac0
    g = grefrac
  }else if (flag == 2) {
    net_send(refrac*p, 3)
    vrev = vrefrac1
  }else if (flag == 3) {
    g = 0
  }else if (flag == 4) {
    WATCH (v > thresh) 1
  }
: printf("Event at t %f:  flag %f  new vrev %f  new g %f\n", t, flag, vrev, g)
}

