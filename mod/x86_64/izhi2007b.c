/* Created by Language version: 7.7.0 */
/* VECTORIZED */
#define NRN_VECTORIZED 1
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
 
#define nrn_init _nrn_init__Izhi2007b
#define _nrn_initial _nrn_initial__Izhi2007b
#define nrn_cur _nrn_cur__Izhi2007b
#define _nrn_current _nrn_current__Izhi2007b
#define nrn_jacob _nrn_jacob__Izhi2007b
#define nrn_state _nrn_state__Izhi2007b
#define _net_receive _net_receive__Izhi2007b 
 
#define _threadargscomma_ _p, _ppvar, _thread, _nt,
#define _threadargsprotocomma_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt,
#define _threadargs_ _p, _ppvar, _thread, _nt
#define _threadargsproto_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 /* Thread safe. No static _p or _ppvar. */
 
#define t _nt->_t
#define dt _nt->_dt
#define C _p[0]
#define C_columnindex 0
#define k _p[1]
#define k_columnindex 1
#define vr _p[2]
#define vr_columnindex 2
#define vt _p[3]
#define vt_columnindex 3
#define vpeak _p[4]
#define vpeak_columnindex 4
#define a _p[5]
#define a_columnindex 5
#define b _p[6]
#define b_columnindex 6
#define c _p[7]
#define c_columnindex 7
#define d _p[8]
#define d_columnindex 8
#define Iin _p[9]
#define Iin_columnindex 9
#define celltype _p[10]
#define celltype_columnindex 10
#define alive _p[11]
#define alive_columnindex 11
#define cellid _p[12]
#define cellid_columnindex 12
#define i _p[13]
#define i_columnindex 13
#define u _p[14]
#define u_columnindex 14
#define delta _p[15]
#define delta_columnindex 15
#define t0 _p[16]
#define t0_columnindex 16
#define derivtype _p[17]
#define derivtype_columnindex 17
#define v _p[18]
#define v_columnindex 18
#define _g _p[19]
#define _g_columnindex 19
#define _tsav _p[20]
#define _tsav_columnindex 20
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
 static Datum* _extcall_thread;
 static Prop* _extcall_prop;
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
 _extcall_prop = _prop;
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
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "vr", "mV",
 "vt", "mV",
 "vpeak", "mV",
 "i", "nA",
 "u", "mV",
 0,0
};
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
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
   if (_prop) { _nrn_free_watch(_prop->dparam, 3, 8);}
   destroy_point_process(_vptr);
}
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"Izhi2007b",
 "C",
 "k",
 "vr",
 "vt",
 "vpeak",
 "a",
 "b",
 "c",
 "d",
 "Iin",
 "celltype",
 "alive",
 "cellid",
 0,
 "i",
 "u",
 "delta",
 "t0",
 "derivtype",
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
 	_p = nrn_prop_data_alloc(_mechtype, 21, _prop);
 	/*initialize range parameters*/
 	C = 1;
 	k = 0.7;
 	vr = -60;
 	vt = -40;
 	vpeak = 35;
 	a = 0.03;
 	b = -2;
 	c = -50;
 	d = 100;
 	Iin = 0;
 	celltype = 1;
 	alive = 1;
 	cellid = -1;
  }
 	_prop->param = _p;
 	_prop->param_size = 21;
  if (!nrn_point_prop_) {
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 11, _prop);
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

 void _izhi2007b_reg() {
	int _vectorized = 1;
  _initlists();
 	_pointtype = point_register_mech(_mechanism,
	 nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init,
	 hoc_nrnpointerindex, 1,
	 _hoc_create_pnt, _hoc_destroy_pnt, _member_func);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 21, 11);
  hoc_reg_watch_allocate(_mechtype, _watch_alloc);
  hoc_register_dparam_semantics(_mechtype, 0, "area");
  hoc_register_dparam_semantics(_mechtype, 1, "pntproc");
  hoc_register_dparam_semantics(_mechtype, 2, "netsend");
  hoc_register_dparam_semantics(_mechtype, 3, "watch");
  hoc_register_dparam_semantics(_mechtype, 4, "watch");
  hoc_register_dparam_semantics(_mechtype, 5, "watch");
  hoc_register_dparam_semantics(_mechtype, 6, "watch");
  hoc_register_dparam_semantics(_mechtype, 7, "watch");
  hoc_register_dparam_semantics(_mechtype, 8, "watch");
  hoc_register_dparam_semantics(_mechtype, 9, "watch");
  hoc_register_dparam_semantics(_mechtype, 10, "watch");
 add_nrn_has_net_event(_mechtype);
 pnt_receive[_mechtype] = _net_receive;
 pnt_receive_size[_mechtype] = 1;
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 Izhi2007b /Users/katedoxey/Desktop/research/projects/tinnitus model/code/DCN_model/mod/izhi2007b.mod\n");
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
 	double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
	_thread= (Datum*)0; _nt = (NrnThread*)_pnt->_vnt;
 	_p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
	v = NODEV(_pnt->node);
	return  ( v ) - ( ( vpeak - 0.1 * u ) ) ;
}
 
static double _watch2_cond(Point_process* _pnt) {
 	double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
	_thread= (Datum*)0; _nt = (NrnThread*)_pnt->_vnt;
 	_p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
	v = NODEV(_pnt->node);
	return  ( v ) - ( ( vpeak + 0.1 * u ) ) ;
}
 
static double _watch3_cond(Point_process* _pnt) {
 	double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
	_thread= (Datum*)0; _nt = (NrnThread*)_pnt->_vnt;
 	_p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
	v = NODEV(_pnt->node);
	return  ( v ) - ( vpeak ) ;
}
 
static double _watch4_cond(Point_process* _pnt) {
 	double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
	_thread= (Datum*)0; _nt = (NrnThread*)_pnt->_vnt;
 	_p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
	v = NODEV(_pnt->node);
	return  ( v ) - ( - 65.0 ) ;
}
 
static double _watch5_cond(Point_process* _pnt) {
 	double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
	_thread= (Datum*)0; _nt = (NrnThread*)_pnt->_vnt;
 	_p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
	v = NODEV(_pnt->node);
	return  -( ( v ) - ( - 65.0 ) ) ;
}
 
static double _watch6_cond(Point_process* _pnt) {
 	double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
	_thread= (Datum*)0; _nt = (NrnThread*)_pnt->_vnt;
 	_p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
	v = NODEV(_pnt->node);
	return  ( v ) - ( d ) ;
}
 
static double _watch7_cond(Point_process* _pnt) {
 	double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
	_thread= (Datum*)0; _nt = (NrnThread*)_pnt->_vnt;
 	_p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
	v = NODEV(_pnt->node);
	return  -( ( v ) - ( d ) ) ;
}
 
static void _net_receive (Point_process* _pnt, double* _args, double _lflag) 
{  double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   int _watch_rm = 0;
   _thread = (Datum*)0; _nt = (NrnThread*)_pnt->_vnt;   _p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
  if (_tsav > t){ extern char* hoc_object_name(); hoc_execerror(hoc_object_name(_pnt->ob), ":Event arrived out of order. Must call ParallelContext.set_maxstep AFTER assigning minimum NetCon.delay");}
 _tsav = t;  v = NODEV(_pnt->node);
   if (_lflag == 1. ) {*(_tqitem) = 0;}
 {
   if ( _lflag  == 1.0 ) {
     if ( celltype  == 4.0 ) {
         _nrn_watch_activate(_watch_array, _watch1_cond, 1, _pnt, _watch_rm++, 2.0);
 }
     else if ( celltype  == 6.0 ) {
         _nrn_watch_activate(_watch_array, _watch2_cond, 2, _pnt, _watch_rm++, 2.0);
 }
     else {
         _nrn_watch_activate(_watch_array, _watch3_cond, 3, _pnt, _watch_rm++, 2.0);
 }
     if ( celltype  == 6.0  || celltype  == 7.0 ) {
         _nrn_watch_activate(_watch_array, _watch4_cond, 4, _pnt, _watch_rm++, 3.0);
   _nrn_watch_activate(_watch_array, _watch5_cond, 5, _pnt, _watch_rm++, 4.0);
 }
     if ( celltype  == 5.0 ) {
         _nrn_watch_activate(_watch_array, _watch6_cond, 6, _pnt, _watch_rm++, 3.0);
   _nrn_watch_activate(_watch_array, _watch7_cond, 7, _pnt, _watch_rm++, 4.0);
 }
     v = vr ;
     }
   else if ( _lflag  == 2.0 ) {
     if ( alive ) {
       net_event ( _pnt, t ) ;
       }
     if ( celltype  == 4.0 ) {
       v = c + 0.04 * u ;
       if ( ( u + d ) < 670.0 ) {
         u = u + d ;
         }
       else {
         u = 670.0 ;
         }
       }
     else if ( celltype  == 5.0 ) {
       v = c ;
       }
     else if ( celltype  == 6.0 ) {
       v = c - 0.1 * u ;
       u = u + d ;
       }
     else {
       v = c ;
       u = u + d ;
       }
     }
   else if ( _lflag  == 3.0 ) {
     if ( celltype  == 5.0 ) {
       derivtype = 1.0 ;
       }
     else if ( celltype  == 6.0 ) {
       b = 0.0 ;
       }
     else if ( celltype  == 7.0 ) {
       b = 2.0 ;
       }
     }
   else if ( _lflag  == 4.0 ) {
     if ( celltype  == 5.0 ) {
       derivtype = 2.0 ;
       }
     else if ( celltype  == 6.0 ) {
       b = 15.0 ;
       }
     else if ( celltype  == 7.0 ) {
       b = 10.0 ;
       }
     }
   } 
 NODEV(_pnt->node) = v;
 }
 
static void _watch_alloc(Datum* _ppvar) {
  Point_process* _pnt = (Point_process*)_ppvar[1]._pvoid;
   _nrn_watch_allocate(_watch_array, _watch1_cond, 1, _pnt, 2.0);
   _nrn_watch_allocate(_watch_array, _watch2_cond, 2, _pnt, 2.0);
   _nrn_watch_allocate(_watch_array, _watch3_cond, 3, _pnt, 2.0);
   _nrn_watch_allocate(_watch_array, _watch4_cond, 4, _pnt, 3.0);
   _nrn_watch_allocate(_watch_array, _watch5_cond, 5, _pnt, 4.0);
   _nrn_watch_allocate(_watch_array, _watch6_cond, 6, _pnt, 3.0);
   _nrn_watch_allocate(_watch_array, _watch7_cond, 7, _pnt, 4.0);
 }


static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
 {
   u = 0.0 ;
   derivtype = 2.0 ;
   net_send ( _tqitem, (double*)0, _ppvar[1]._pvoid, t +  0.0 , 1.0 ) ;
   }

}
}

static void nrn_init(NrnThread* _nt, _Memb_list* _ml, int _type){
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
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
 initmodel(_p, _ppvar, _thread, _nt);
}
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   delta = t - t0 ;
   if ( celltype < 5.0 ) {
     u = u + delta * a * ( b * ( v - vr ) - u ) ;
     }
   else {
     if ( celltype  == 5.0 ) {
       if ( v < d ) {
         u = u + delta * a * ( 0.0 - u ) ;
         }
       else {
         u = u + delta * a * ( ( 0.025 * ( v - d ) * ( v - d ) * ( v - d ) ) - u ) ;
         }
       }
     if ( celltype  == 6.0 ) {
       if ( v > - 65.0 ) {
         b = 0.0 ;
         }
       else {
         b = 15.0 ;
         }
       u = u + delta * a * ( b * ( v - vr ) - u ) ;
       }
     if ( celltype  == 7.0 ) {
       if ( v > - 65.0 ) {
         b = 2.0 ;
         }
       else {
         b = 10.0 ;
         }
       u = u + delta * a * ( b * ( v - vr ) - u ) ;
       }
     }
   t0 = t ;
   i = - ( k * ( v - vr ) * ( v - vt ) - u + Iin ) / C / 1000.0 ;
   }
 _current += i;

} return _current;
}

static void nrn_cur(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
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
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
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
 
}
 
}

static void nrn_jacob(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
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
 
}
 
}

static void nrn_state(NrnThread* _nt, _Memb_list* _ml, int _type) {

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/Users/katedoxey/Desktop/research/projects/tinnitus model/code/DCN_model/mod/izhi2007b.mod";
static const char* nmodl_file_text = 
  "COMMENT\n"
  "\n"
  "A \"simple\" implementation of the Izhikevich neuron.\n"
  "Equations and parameter values are taken from\n"
  "  Izhikevich EM (2007).\n"
  "  \"Dynamical systems in neuroscience\"\n"
  "  MIT Press\n"
  "\n"
  "Equation for synaptic inputs taken from\n"
  "  Izhikevich EM, Edelman GM (2008).\n"
  "  \"Large-scale model of mammalian thalamocortical systems.\" \n"
  "  PNAS 105(9) 3593-3598.\n"
  "\n"
  "Example usage (in Python):\n"
  "  from neuron import h\n"
  "  sec = h.Section(name=sec) # section will be used to calculate v\n"
  "  izh = h.Izhi2007b(0.5)\n"
  "  def initiz () : sec.v=-60\n"
  "  fih=h.FInitializeHandler(initz)\n"
  "  izh.Iin = 70  # current clamp\n"
  "\n"
  "Cell types available are based on Izhikevich, 2007 book:\n"
  "    1. RS - Layer 5 regular spiking pyramidal cell (fig 8.12 from 2007 book)\n"
  "    2. IB - Layer 5 intrinsically bursting cell (fig 8.19 from 2007 book)\n"
  "    3. CH - Cat primary visual cortex chattering cell (fig 8.23 from 2007 book)\n"
  "    4. LTS - Rat barrel cortex Low-threshold  spiking interneuron (fig 8.25 from 2007 book)\n"
  "    5. FS - Rat visual cortex layer 5 fast-spiking interneuron (fig 8.27 from 2007 book)\n"
  "    6. TC - Cat dorsal LGN thalamocortical (TC) cell (fig 8.31 from 2007 book)\n"
  "    7. RTN - Rat reticular thalamic nucleus (RTN) cell  (fig 8.32 from 2007 book)\n"
  "\n"
  "ENDCOMMENT\n"
  "\n"
  ": Declare name of object and variables\n"
  "NEURON {\n"
  "  POINT_PROCESS Izhi2007b\n"
  "  RANGE C, k, vr, vt, vpeak, u, a, b, c, d, Iin, celltype, alive, cellid, verbose, derivtype, delta, t0\n"
  "  NONSPECIFIC_CURRENT i\n"
  "}\n"
  "\n"
  ": Specify units that have physiological interpretations (NB: ms is already declared)\n"
  "UNITS {\n"
  "  (mV) = (millivolt)\n"
  "  (uM) = (micrometer)\n"
  "}\n"
  "\n"
  ": Parameters from Izhikevich 2007, MIT Press for regular spiking pyramidal cell\n"
  "PARAMETER {\n"
  "  C = 1 : Capacitance\n"
  "  k = 0.7\n"
  "  vr = -60 (mV) : Resting membrane potential\n"
  "  vt = -40 (mV) : Membrane threhsold\n"
  "  vpeak = 35 (mV) : Peak voltage\n"
  "  a = 0.03\n"
  "  b = -2\n"
  "  c = -50\n"
  "  d = 100\n"
  "  Iin = 0\n"
  "  celltype = 1 : A flag for indicating what kind of cell it is,  used for changing the dynamics slightly (see list of cell types in initial comment).\n"
  "  alive = 1 : A flag for deciding whether or not the cell is alive -- if it's dead, acts normally except it doesn't fire spikes\n"
  "  cellid = -1 : A parameter for storing the cell ID, if required (useful for diagnostic information)\n"
  "}\n"
  "\n"
  ": Variables used for internal calculations\n"
  "ASSIGNED {\n"
  "  v (mV)\n"
  "  i (nA)\n"
  "  u (mV) : Slow current/recovery variable\n"
  "  delta\n"
  "  t0\n"
  "  derivtype\n"
  "}\n"
  "\n"
  ": Initial conditions\n"
  "INITIAL {\n"
  "  u = 0.0\n"
  "  derivtype=2\n"
  "  net_send(0,1) : Required for the WATCH statement to be active; v=vr initialization done there\n"
  "}\n"
  "\n"
  ": Define neuron dynamics\n"
  "BREAKPOINT {\n"
  "  delta = t-t0 : Find time difference\n"
  "  if (celltype<5) {\n"
  "    u = u + delta*a*(b*(v-vr)-u) : Calculate recovery variable\n"
  "  }\n"
  "  else {\n"
  "     : For FS neurons, include nonlinear U(v): U(v) = 0 when v<vb ; U(v) = 0.025(v-vb) when v>=vb (d=vb=-55)\n"
  "     if (celltype==5) {\n"
  "       if (v<d) { \n"
  "        u = u + delta*a*(0-u)\n"
  "       }\n"
  "       else { \n"
  "        u = u + delta*a*((0.025*(v-d)*(v-d)*(v-d))-u)\n"
  "       }\n"
  "     }\n"
  "\n"
  "     : For TC neurons, reset b\n"
  "     if (celltype==6) {\n"
  "       if (v>-65) {b=0}\n"
  "       else {b=15}\n"
  "       u = u + delta*a*(b*(v-vr)-u) : Calculate recovery variable\n"
  "     }\n"
  "     \n"
  "     : For TRN neurons, reset b\n"
  "     if (celltype==7) {\n"
  "       if (v>-65) {b=2}\n"
  "       else {b=10}\n"
  "       u = u + delta*a*(b*(v-vr)-u) : Calculate recovery variable\n"
  "     }\n"
  "  }\n"
  "\n"
  "  t0=t : Reset last time so delta can be calculated in the next time step\n"
  "  i = -(k*(v-vr)*(v-vt) - u + Iin)/C/1000\n"
  "}\n"
  "\n"
  ": Input received\n"
  "NET_RECEIVE (w) {\n"
  "  : Check if spike occurred\n"
  "  if (flag == 1) { : Fake event from INITIAL block\n"
  "    if (celltype == 4) { : LTS cell\n"
  "      WATCH (v>(vpeak-0.1*u)) 2 : Check if threshold has been crossed, and if so, set flag=2     \n"
  "    } else if (celltype == 6) { : TC cell\n"
  "      WATCH (v>(vpeak+0.1*u)) 2 \n"
  "    } else { : default for all other types\n"
  "      WATCH (v>vpeak) 2 \n"
  "    }\n"
  "    : additional WATCHfulness\n"
  "    if (celltype==6 || celltype==7) {\n"
  "      WATCH (v> -65) 3 : change b param\n"
  "      WATCH (v< -65) 4 : change b param\n"
  "    }\n"
  "    if (celltype==5) {\n"
  "      WATCH (v> d) 3  : going up\n"
  "      WATCH (v< d) 4  : coming down\n"
  "    }\n"
  "    v = vr  : initialization can be done here\n"
  "  : FLAG 2 Event created by WATCH statement -- threshold crossed for spiking\n"
  "  } else if (flag == 2) { \n"
  "    if (alive) {net_event(t)} : Send spike event if the cell is alive\n"
  "    : For LTS neurons\n"
  "    if (celltype == 4) {\n"
  "      v = c+0.04*u : Reset voltage\n"
  "      if ((u+d)<670) {u=u+d} : Reset recovery variable\n"
  "      else {u=670} \n"
  "     }  \n"
  "    : For FS neurons (only update v)\n"
  "    else if (celltype == 5) {\n"
  "      v = c : Reset voltage\n"
  "     }  \n"
  "    : For TC neurons (only update v)\n"
  "    else if (celltype == 6) {\n"
  "      v = c-0.1*u : Reset voltage\n"
  "      u = u+d : Reset recovery variable\n"
  "     }  else {: For RS, IB and CH neurons, and RTN\n"
  "      v = c : Reset voltage\n"
  "      u = u+d : Reset recovery variable\n"
  "     }\n"
  "  : FLAG 3 Event created by WATCH statement -- v exceeding set point for param reset\n"
  "  } else if (flag == 3) { \n"
  "    : For TC neurons \n"
  "    if (celltype == 5)        { derivtype = 1 : if (v>d) u'=a*((0.025*(v-d)*(v-d)*(v-d))-u)\n"
  "    } else if (celltype == 6) { b=0\n"
  "    } else if (celltype == 7) { b=2 \n"
  "    }\n"
  "  : FLAG 4 Event created by WATCH statement -- v dropping below a setpoint for param reset\n"
  "  } else if (flag == 4) { \n"
  "    if (celltype == 5)        { derivtype = 2  : if (v<d) u==a*(0-u)\n"
  "    } else if (celltype == 6) { b=15\n"
  "    } else if (celltype == 7) { b=10\n"
  "    }\n"
  "  }\n"
  "}\n"
  ;
#endif
