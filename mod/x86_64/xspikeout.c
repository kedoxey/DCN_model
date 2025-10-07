/* Created by Language version: 7.7.0 */
/* NOT VECTORIZED */
#define NRN_VECTORIZED 0
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mech_api.h"
#undef PI
#define nil 0
#include "md1redef.h"
#include "section.h"
#include "nrniv_mf.h"
#include "md2redef.h"
 
#if METHOD3
extern int _method3;
#endif

#if !NRNGPU
#undef exp
#define exp hoc_Exp
extern double hoc_Exp(double);
#endif
 
#define nrn_init _nrn_init__XSpikeOut
#define _nrn_initial _nrn_initial__XSpikeOut
#define nrn_cur _nrn_cur__XSpikeOut
#define _nrn_current _nrn_current__XSpikeOut
#define nrn_jacob _nrn_jacob__XSpikeOut
#define nrn_state _nrn_state__XSpikeOut
#define _net_receive _net_receive__XSpikeOut 
 
#define _threadargscomma_ /**/
#define _threadargsprotocomma_ /**/
#define _threadargs_ /**/
#define _threadargsproto_ /**/
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 static double *_p; static Datum *_ppvar;
 
#define t nrn_threads->_t
#define dt nrn_threads->_dt
#define i _p[0]
#define i_columnindex 0
#define vrev _p[1]
#define vrev_columnindex 1
#define g _p[2]
#define g_columnindex 2
#define _g _p[3]
#define _g_columnindex 3
#define _tsav _p[4]
#define _tsav_columnindex 4
#define _nd_area  *_ppvar[0]._pval
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 
#if defined(__cplusplus)
extern "C" {
#endif
 static int hoc_nrnpointerindex =  -1;
 /* external NEURON variables */
 /* declaration of user functions */
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_prop_size(int, int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
extern Memb_func* memb_func;
 
#define NMODL_TEXT 1
#if NMODL_TEXT
static const char* nmodl_file_text;
static const char* nmodl_filename;
extern void hoc_reg_nmodl_text(int, const char*);
extern void hoc_reg_nmodl_filename(int, const char*);
#endif

 extern Prop* nrn_point_prop_;
 static int _pointtype;
 static void* _hoc_create_pnt(Object* _ho) { void* create_point_process(int, Object*);
 return create_point_process(_pointtype, _ho);
}
 static void _hoc_destroy_pnt(void*);
 static double _hoc_loc_pnt(void* _vptr) {double loc_point_process(int, void*);
 return loc_point_process(_pointtype, _vptr);
}
 static double _hoc_has_loc(void* _vptr) {double has_loc_point(void*);
 return has_loc_point(_vptr);
}
 static double _hoc_get_loc_pnt(void* _vptr) {
 double get_loc_point_process(void*); return (get_loc_point_process(_vptr));
}
 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _p = _prop->param; _ppvar = _prop->dparam;
 }
 static void _hoc_setdata(void* _vptr) { Prop* _prop;
 _prop = ((Point_process*)_vptr)->_prop;
   _setdata(_prop);
 }
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 0,0
};
 static Member_func _member_func[] = {
 "loc", _hoc_loc_pnt,
 "has_loc", _hoc_has_loc,
 "get_loc", _hoc_get_loc_pnt,
 0, 0
};
 /* declare global and static user variables */
#define grefrac grefrac_XSpikeOut
 double grefrac = 100;
#define p p_XSpikeOut
 double p = 0;
#define refrac refrac_XSpikeOut
 double refrac = 3;
#define thresh thresh_XSpikeOut
 double thresh = -55;
#define vrefrac1 vrefrac1_XSpikeOut
 double vrefrac1 = 10;
#define vrefrac0 vrefrac0_XSpikeOut
 double vrefrac0 = -60;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "thresh_XSpikeOut", "millivolt",
 "refrac_XSpikeOut", "ms",
 "vrefrac0_XSpikeOut", "millivolt",
 "vrefrac1_XSpikeOut", "millivolt",
 "grefrac_XSpikeOut", "microsiemens",
 "i", "nanoamp",
 "vrev", "millivolt",
 "g", "microsiemens",
 0,0
};
 static double v = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "thresh_XSpikeOut", &thresh_XSpikeOut,
 "refrac_XSpikeOut", &refrac_XSpikeOut,
 "vrefrac0_XSpikeOut", &vrefrac0_XSpikeOut,
 "vrefrac1_XSpikeOut", &vrefrac1_XSpikeOut,
 "p_XSpikeOut", &p_XSpikeOut,
 "grefrac_XSpikeOut", &grefrac_XSpikeOut,
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(Prop*);
static void  nrn_init(NrnThread*, _Memb_list*, int);
static void nrn_state(NrnThread*, _Memb_list*, int);
 static void nrn_cur(NrnThread*, _Memb_list*, int);
static void  nrn_jacob(NrnThread*, _Memb_list*, int);
 
#define _watch_array _ppvar + 3 
 static void _watch_alloc(Datum*);
 extern void hoc_reg_watch_allocate(int, void(*)(Datum*)); static void _hoc_destroy_pnt(void* _vptr) {
   Prop* _prop = ((Point_process*)_vptr)->_prop;
   if (_prop) { _nrn_free_watch(_prop->dparam, 3, 2);}
   destroy_point_process(_vptr);
}
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"XSpikeOut",
 0,
 "i",
 "vrev",
 "g",
 0,
 0,
 0};
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
  if (nrn_point_prop_) {
	_prop->_alloc_seq = nrn_point_prop_->_alloc_seq;
	_p = nrn_point_prop_->param;
	_ppvar = nrn_point_prop_->dparam;
 }else{
 	_p = nrn_prop_data_alloc(_mechtype, 5, _prop);
 	/*initialize range parameters*/
  }
 	_prop->param = _p;
 	_prop->param_size = 5;
  if (!nrn_point_prop_) {
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 5, _prop);
  }
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 
}
 static void _initlists();
 
#define _tqitem &(_ppvar[2]._pvoid)
 static void _net_receive(Point_process*, double*, double);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _xspikeout_reg() {
	int _vectorized = 0;
  _initlists();
 	_pointtype = point_register_mech(_mechanism,
	 nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init,
	 hoc_nrnpointerindex, 0,
	 _hoc_create_pnt, _hoc_destroy_pnt, _member_func);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 5, 5);
  hoc_reg_watch_allocate(_mechtype, _watch_alloc);
  hoc_register_dparam_semantics(_mechtype, 0, "area");
  hoc_register_dparam_semantics(_mechtype, 1, "pntproc");
  hoc_register_dparam_semantics(_mechtype, 2, "netsend");
  hoc_register_dparam_semantics(_mechtype, 3, "watch");
  hoc_register_dparam_semantics(_mechtype, 4, "watch");
 add_nrn_has_net_event(_mechtype);
 pnt_receive[_mechtype] = _net_receive;
 pnt_receive_size[_mechtype] = 1;
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 XSpikeOut /home/kedoxey/DCN_model/mod/xspikeout.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
 
static double _watch1_cond(Point_process* _pnt) {
  	_p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
	v = NODEV(_pnt->node);
	return  ( v ) - ( thresh ) ;
}
 
static void _net_receive (Point_process* _pnt, double* _args, double _lflag) 
{   int _watch_rm = 0;
    _p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
  if (_tsav > t){ extern char* hoc_object_name(); hoc_execerror(hoc_object_name(_pnt->ob), ":Event arrived out of order. Must call ParallelContext.set_maxstep AFTER assigning minimum NetCon.delay");}
 _tsav = t;   if (_lflag == 1. ) {*(_tqitem) = 0;}
 {
   if ( _lflag  == 1.0 ) {
     net_event ( _pnt, t ) ;
     net_send ( _tqitem, _args, _pnt, t +  refrac * ( 1.0 - p ) , 2.0 ) ;
     vrev = vrefrac0 ;
     g = grefrac ;
     }
   else if ( _lflag  == 2.0 ) {
     net_send ( _tqitem, _args, _pnt, t +  refrac * p , 3.0 ) ;
     vrev = vrefrac1 ;
     }
   else if ( _lflag  == 3.0 ) {
     g = 0.0 ;
     }
   else if ( _lflag  == 4.0 ) {
       _nrn_watch_activate(_watch_array, _watch1_cond, 1, _pnt, _watch_rm++, 1.0);
 }
   } }
 
static void _watch_alloc(Datum* _ppvar) {
  Point_process* _pnt = (Point_process*)_ppvar[1]._pvoid;
   _nrn_watch_allocate(_watch_array, _watch1_cond, 1, _pnt, 1.0);
 }


static void initmodel() {
  int _i; double _save;_ninits++;
{
 {
   if ( refrac < 0.0 ) {
     refrac = 0.0 ;
     }
   if ( grefrac < 0.0 ) {
     grefrac = 0.0 ;
     }
   if ( p < 0.0 ) {
     p = 0.0 ;
     }
   if ( p > 1.0 ) {
     p = 1.0 ;
     }
   g = 0.0 ;
   vrev = v ;
   net_send ( _tqitem, (double*)0, _ppvar[1]._pvoid, t +  0.0 , 4.0 ) ;
   }

}
}

static void nrn_init(NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _tsav = -1e20;
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v = _v;
 initmodel();
}}

static double _nrn_current(double _v){double _current=0.;v=_v;{ {
   i = g * ( v - vrev ) ;
   }
 _current += i;

} return _current;
}

static void nrn_cur(NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 _g = _nrn_current(_v + .001);
 	{ _rhs = _nrn_current(_v);
 	}
 _g = (_g - _rhs)/.001;
 _g *=  1.e2/(_nd_area);
 _rhs *= 1.e2/(_nd_area);
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}}

static void nrn_jacob(NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml];
#if CACHEVEC
  if (use_cachevec) {
	VEC_D(_ni[_iml]) += _g;
  }else
#endif
  {
     _nd = _ml->_nodelist[_iml];
	NODED(_nd) += _g;
  }
 
}}

static void nrn_state(NrnThread* _nt, _Memb_list* _ml, int _type){

}

static void terminal(){}

static void _initlists() {
 int _i; static int _first = 1;
  if (!_first) return;
_first = 0;
}

#if NMODL_TEXT
static const char* nmodl_filename = "/home/kedoxey/DCN_model/mod/xspikeout.mod";
static const char* nmodl_file_text = 
  "NEURON {\n"
  ":  POINT_PROCESS SpikeOut\n"
  "  POINT_PROCESS XSpikeOut\n"
  ":  GLOBAL thresh, refrac, vrefrac, grefrac\n"
  "  GLOBAL thresh, refrac, p, vrefrac0, vrefrac1, grefrac\n"
  "  NONSPECIFIC_CURRENT i\n"
  "  RANGE g, vrev\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "  thresh = -55 (millivolt)\n"
  "  refrac = 3 (ms)\n"
  ":  vrefrac = -60 (millivolt)\n"
  "  vrefrac0 = -60 (millivolt)\n"
  "  vrefrac1 = 10 (millivolt)\n"
  "  p = 0 : fraction of refractory interval \n"
  "        : during which vrev == vrefrac1\n"
  "        : if p = 0, vrev = vrefrac0 throughout the entire refractory interval\n"
  "        : if p = 1, vrev = vrefrac1 throughout the entire refractory interval\n"
  "  grefrac = 100 (microsiemens) :clamp to vrefrac\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "  i (nanoamp)\n"
  "  v (millivolt)\n"
  "  vrev (millivolt)\n"
  "  g (microsiemens)\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  ":  net_send(0, 3) : because there were three actions that had to occur--\n"
  "    : initialization, starting a spike, and ending a spike.\n"
  "    : These were controlled by self events with a flag variable \n"
  "    : that could have one of three values--\n"
  "    : 1 meant that v crossed spike threshold, so it is time to enter the \"spike sequence\"\n"
  "    : 2 meant it is time to end the spike sequence\n"
  "    : 3 meant \"execute the WATCH statement\"\n"
  "    : Now there are four actions--\n"
  "    : initialization, starting a spike, changing vrev, and ending a spike.\n"
  "  if (refrac<0) { : force refrac >=0\n"
  "    refrac = 0\n"
  "  }\n"
  "  if (grefrac<0) { : force grefrac >=0\n"
  "    grefrac = 0\n"
  "  }\n"
  "  if (p<0) { : force 0<=p<=1\n"
  "    p = 0\n"
  "  }\n"
  "  if (p>1) {\n"
  "    p = 1\n"
  "  }\n"
  "  g = 0\n"
  "  vrev = v\n"
  ": printf(\"INITIAL block:  t %f  vrev %f  g %f\\n\", t, vrev, g)\n"
  "  net_send(0, 4)\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  ":  i = g*(v - vrefrac)\n"
  "  i = g*(v - vrev)\n"
  "}\n"
  "\n"
  "COMMENT\n"
  "NET_RECEIVE(w) {\n"
  "  if (flag == 1) {\n"
  "    net_event(t)\n"
  "    net_send(refrac, 2)\n"
  "    v = vrefrac\n"
  "    g = grefrac\n"
  "  }else if (flag == 2) {\n"
  "    g = 0\n"
  "  }else if (flag == 3) {\n"
  "    WATCH (v > thresh) 1\n"
  "  }  \n"
  "}\n"
  "ENDCOMMENT\n"
  "\n"
  "NET_RECEIVE(w) {\n"
  "  if (flag == 1) {\n"
  "    net_event(t)\n"
  "    net_send(refrac*(1-p), 2)\n"
  ":    v = vrefrac\n"
  "    vrev = vrefrac0\n"
  "    g = grefrac\n"
  "  }else if (flag == 2) {\n"
  "    net_send(refrac*p, 3)\n"
  "    vrev = vrefrac1\n"
  "  }else if (flag == 3) {\n"
  "    g = 0\n"
  "  }else if (flag == 4) {\n"
  "    WATCH (v > thresh) 1\n"
  "  }\n"
  ": printf(\"Event at t %f:  flag %f  new vrev %f  new g %f\\n\", t, flag, vrev, g)\n"
  "}\n"
  "\n"
  ;
#endif
