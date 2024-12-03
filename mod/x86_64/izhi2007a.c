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
 
#define nrn_init _nrn_init__Izhi2007a
#define _nrn_initial _nrn_initial__Izhi2007a
#define nrn_cur _nrn_cur__Izhi2007a
#define _nrn_current _nrn_current__Izhi2007a
#define nrn_jacob _nrn_jacob__Izhi2007a
#define nrn_state _nrn_state__Izhi2007a
#define _net_receive _net_receive__Izhi2007a 
#define useverbose useverbose__Izhi2007a 
 
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
#define tauAMPA _p[10]
#define tauAMPA_columnindex 10
#define tauNMDA _p[11]
#define tauNMDA_columnindex 11
#define tauGABAA _p[12]
#define tauGABAA_columnindex 12
#define tauGABAB _p[13]
#define tauGABAB_columnindex 13
#define tauOpsin _p[14]
#define tauOpsin_columnindex 14
#define celltype _p[15]
#define celltype_columnindex 15
#define alive _p[16]
#define alive_columnindex 16
#define cellid _p[17]
#define cellid_columnindex 17
#define verbose _p[18]
#define verbose_columnindex 18
#define factor _p[19]
#define factor_columnindex 19
#define eventflag _p[20]
#define eventflag_columnindex 20
#define V _p[21]
#define V_columnindex 21
#define u _p[22]
#define u_columnindex 22
#define gAMPA _p[23]
#define gAMPA_columnindex 23
#define gNMDA _p[24]
#define gNMDA_columnindex 24
#define gGABAA _p[25]
#define gGABAA_columnindex 25
#define gGABAB _p[26]
#define gGABAB_columnindex 26
#define gOpsin _p[27]
#define gOpsin_columnindex 27
#define I _p[28]
#define I_columnindex 28
#define delta _p[29]
#define delta_columnindex 29
#define t0 _p[30]
#define t0_columnindex 30
#define _g _p[31]
#define _g_columnindex 31
#define _tsav _p[32]
#define _tsav_columnindex 32
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
 static double _hoc_useverbose(void*);
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
 "useverbose", _hoc_useverbose,
 0, 0
};
 /* declare global and static user variables */
#define Vpre Vpre_Izhi2007a
 double Vpre = 0;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "vr", "mV",
 "vt", "mV",
 "vpeak", "mV",
 "tauAMPA", "ms",
 "tauNMDA", "ms",
 "tauGABAA", "ms",
 "tauGABAB", "ms",
 "tauOpsin", "ms",
 "V", "mV",
 "u", "mV",
 0,0
};
 static double v = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "Vpre_Izhi2007a", &Vpre_Izhi2007a,
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
   if (_prop) { _nrn_free_watch(_prop->dparam, 3, 4);}
   destroy_point_process(_vptr);
}
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"Izhi2007a",
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
 "tauAMPA",
 "tauNMDA",
 "tauGABAA",
 "tauGABAB",
 "tauOpsin",
 "celltype",
 "alive",
 "cellid",
 "verbose",
 0,
 "factor",
 "eventflag",
 "V",
 "u",
 "gAMPA",
 "gNMDA",
 "gGABAA",
 "gGABAB",
 "gOpsin",
 "I",
 "delta",
 "t0",
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
 	_p = nrn_prop_data_alloc(_mechtype, 33, _prop);
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
 	tauAMPA = 5;
 	tauNMDA = 150;
 	tauGABAA = 6;
 	tauGABAB = 150;
 	tauOpsin = 50;
 	celltype = 1;
 	alive = 1;
 	cellid = -1;
 	verbose = 0;
  }
 	_prop->param = _p;
 	_prop->param_size = 33;
  if (!nrn_point_prop_) {
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 7, _prop);
  }
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 
}
 static void _initlists();
 
#define _tqitem &(_ppvar[2]._pvoid)
 static void _net_receive(Point_process*, double*, double);
 static void _net_init(Point_process*, double*, double);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _izhi2007a_reg() {
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
  hoc_register_prop_size(_mechtype, 33, 7);
  hoc_reg_watch_allocate(_mechtype, _watch_alloc);
  hoc_register_dparam_semantics(_mechtype, 0, "area");
  hoc_register_dparam_semantics(_mechtype, 1, "pntproc");
  hoc_register_dparam_semantics(_mechtype, 2, "netsend");
  hoc_register_dparam_semantics(_mechtype, 3, "watch");
  hoc_register_dparam_semantics(_mechtype, 4, "watch");
  hoc_register_dparam_semantics(_mechtype, 5, "watch");
  hoc_register_dparam_semantics(_mechtype, 6, "watch");
 add_nrn_has_net_event(_mechtype);
 pnt_receive[_mechtype] = _net_receive;
 pnt_receive_init[_mechtype] = _net_init;
 pnt_receive_size[_mechtype] = 5;
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 Izhi2007a /Users/katedoxey/Desktop/research/projects/tinnitus model/code/DCN_model/mod/izhi2007a.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int useverbose();
 
/*VERBATIM*/
char filename[1000]; // Allocate some memory for the filename
 
static int  useverbose (  ) {
   
/*VERBATIM*/
  #include<stdio.h> // Basic input-output
  verbose = (float) *getarg(1); // Set verbosity -- 0 = none, 1 = events, 2 = events + timesteps
  strcpy(filename, gargstr(2)); // Copy input filename into memory
  return 0; }
 
static double _hoc_useverbose(void* _vptr) {
 double _r;
    _hoc_setdata(_vptr);
 _r = 1.;
 useverbose (  );
 return(_r);
}
 
static double _watch1_cond(Point_process* _pnt) {
  	_p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
	v = NODEV(_pnt->node);
	return  ( V ) - ( vpeak ) ;
}
 
static double _watch2_cond(Point_process* _pnt) {
  	_p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
	v = NODEV(_pnt->node);
	return  ( V ) - ( ( vpeak - 0.1 * u ) ) ;
}
 
static double _watch3_cond(Point_process* _pnt) {
  	_p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
	v = NODEV(_pnt->node);
	return  ( V ) - ( ( vpeak + 0.1 * u ) ) ;
}
 
static void _net_receive (Point_process* _pnt, double* _args, double _lflag) 
{   int _watch_rm = 0;
    _p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
  if (_tsav > t){ extern char* hoc_object_name(); hoc_execerror(hoc_object_name(_pnt->ob), ":Event arrived out of order. Must call ParallelContext.set_maxstep AFTER assigning minimum NetCon.delay");}
 _tsav = t;   if (_lflag == 1. ) {*(_tqitem) = 0;}
 {
   if ( _lflag  == 1.0 ) {
     if ( celltype < 4.0  || celltype  == 5.0  || celltype  == 7.0 ) {
         _nrn_watch_activate(_watch_array, _watch1_cond, 1, _pnt, _watch_rm++, 2.0);
 }
     else if ( celltype  == 4.0 ) {
         _nrn_watch_activate(_watch_array, _watch2_cond, 2, _pnt, _watch_rm++, 2.0);
 }
     else if ( celltype  == 6.0 ) {
         _nrn_watch_activate(_watch_array, _watch3_cond, 3, _pnt, _watch_rm++, 2.0);
 }
     }
   else if ( _lflag  == 2.0 ) {
     if ( alive ) {
       net_event ( _pnt, t ) ;
       }
     if ( celltype < 4.0  || celltype  == 7.0 ) {
       V = c ;
       u = u + d ;
       }
     else if ( celltype  == 4.0 ) {
       V = c + 0.04 * u ;
       if ( ( u + d ) < 670.0 ) {
         u = u + d ;
         }
       else {
         u = 670.0 ;
         }
       }
     else if ( celltype  == 5.0 ) {
       V = c ;
       }
     else if ( celltype  == 6.0 ) {
       V = c - 0.1 * u ;
       u = u + d ;
       }
     gAMPA = 0.0 ;
     gNMDA = 0.0 ;
     gGABAA = 0.0 ;
     gGABAB = 0.0 ;
     gOpsin = 0.0 ;
     }
   else {
     gAMPA = gAMPA + _args[0] ;
     gNMDA = gNMDA + _args[1] ;
     gGABAA = gGABAA + _args[2] ;
     gGABAB = gGABAB + _args[3] ;
     gOpsin = gOpsin + _args[4] ;
     }
   if ( verbose > 0.0 ) {
     eventflag = _lflag ;
     
/*VERBATIM*/
    FILE *outfile; // Declare file object
//if(cellid>=0 && cellid < 300) {
    outfile=fopen(filename,"a"); // Open file for appending
    fprintf(outfile,"t=%8.2f   cell=%6.0f   flag=%1.0f   gAMPA=%8.2f   gNMDA=%8.2f   gGABAA=%8.2f   gGABAB=%8.2f   gOpsin=%8.2f   V=%8.2f   u=%8.2f (event)\n",t, cellid,eventflag,gAMPA,gNMDA,gGABAA,gGABAB,gOpsin,V,u);
    fclose(outfile); // Close file
//}
 }
   } }
 
static void _net_init(Point_process* _pnt, double* _args, double _lflag) {
       _p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
 _args[0] = _args[0] ;
   _args[1] = _args[1] ;
   _args[2] = _args[2] ;
   _args[3] = _args[3] ;
   _args[4] = _args[4] ;
   }
 
static void _watch_alloc(Datum* _ppvar) {
  Point_process* _pnt = (Point_process*)_ppvar[1]._pvoid;
   _nrn_watch_allocate(_watch_array, _watch1_cond, 1, _pnt, 2.0);
   _nrn_watch_allocate(_watch_array, _watch2_cond, 2, _pnt, 2.0);
   _nrn_watch_allocate(_watch_array, _watch3_cond, 3, _pnt, 2.0);
 }


static void initmodel() {
  int _i; double _save;_ninits++;
{
 {
   V = vr ;
   u = 0.0 ;
   t0 = t ;
   gAMPA = 0.0 ;
   gNMDA = 0.0 ;
   gGABAA = 0.0 ;
   gGABAB = 0.0 ;
   gOpsin = 0.0 ;
   I = 0.0 ;
   delta = 0.0 ;
   net_send ( _tqitem, (double*)0, _ppvar[1]._pvoid, t +  0.0 , 1.0 ) ;
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

static double _nrn_current(double _v){double _current=0.;v=_v;{
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
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _nd = _ml->_nodelist[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v=_v;
{
 {
   delta = t - t0 ;
   gAMPA = gAMPA - delta * gAMPA / tauAMPA ;
   gNMDA = gNMDA - delta * gNMDA / tauNMDA ;
   gGABAA = gGABAA - delta * gGABAA / tauGABAA ;
   gGABAB = gGABAB - delta * gGABAB / tauGABAB ;
   gOpsin = gOpsin - delta * gOpsin / tauOpsin ;
   factor = ( ( V + 80.0 ) / 60.0 ) * ( ( V + 80.0 ) / 60.0 ) ;
   I = gAMPA * ( V - 0.0 ) + gNMDA * factor / ( 1.0 + factor ) * ( V - 0.0 ) + gGABAA * ( V + 70.0 ) + gGABAB * ( V + 90.0 ) + gOpsin * ( V - 0.0 ) ;
   Vpre = V ;
   V = V + delta * ( k * ( V - vr ) * ( V - vt ) - u - I + Iin ) / ( C * 100.0 ) ;
   if ( Vpre <= c  && V > vpeak ) {
     V = c + 1.0 ;
     }
   if ( celltype < 5.0 ) {
     u = u + delta * a * ( b * ( V - vr ) - u ) ;
     }
   else {
     if ( celltype  == 5.0 ) {
       if ( V < d ) {
         u = u + delta * a * ( 0.0 - u ) ;
         }
       else {
         u = u + delta * a * ( ( 0.025 * ( V - d ) * ( V - d ) * ( V - d ) ) - u ) ;
         }
       }
     if ( celltype  == 6.0 ) {
       if ( V > - 65.0 ) {
         b = 0.0 ;
         }
       else {
         b = 15.0 ;
         }
       u = u + delta * a * ( b * ( V - vr ) - u ) ;
       }
     if ( celltype  == 7.0 ) {
       if ( V > - 65.0 ) {
         b = 2.0 ;
         }
       else {
         b = 10.0 ;
         }
       u = u + delta * a * ( b * ( V - vr ) - u ) ;
       }
     }
   t0 = t ;
   if ( verbose > 1.0 ) {
     
/*VERBATIM*/
    FILE *outfile; // Declare file object
    outfile=fopen(filename,"a"); // Open file for appending
    fprintf(outfile,"%8.2f   cell=%6.0f   delta=%8.2f   gAMPA=%8.2f   gNMDA=%8.2f   gGABAA=%8.2f   gGABAB=%8.2f   gOpsin=%8.2f   factor=%8.2f   I=%8.2f   V=%8.2f   u=%8.2f (timestep)\n",t,cellid,delta,gAMPA,gNMDA,gGABAA,gGABAB,gOpsin,factor,I,V,u);
    fclose(outfile); // Close file
 }
   }
}}

}

static void terminal(){}

static void _initlists() {
 int _i; static int _first = 1;
  if (!_first) return;
_first = 0;
}

#if NMODL_TEXT
static const char* nmodl_filename = "/Users/katedoxey/Desktop/research/projects/tinnitus model/code/DCN_model/mod/izhi2007a.mod";
static const char* nmodl_file_text = 
  "COMMENT\n"
  "A \"simple\" implementation of the Izhikevich neuron with AMPA, NMDA,\n"
  "GABA_A, and GABA_B receptor dynamics. Equations and parameter values are taken from\n"
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
  "  dummycell = h.Section() # Since Izhi is a point process, it needs to be in a section\n"
  "  izhl = [h.Izhi2007(0.5) for i in range(2)] # Create two new Izhikevich cells\n"
  "  connection = h.NetCon(izhl[0], izhl[1]) # Connect them\n"
  "  izhl[0].Iin = 70  # activate 1 cell\n"
  "\n"
  "Cell types available are based on Izhikevich, 2007 book:\n"
  "    1. RS - Layer 5 regular spiking pyramidal cell (fig 8.12 from 2007 book)\n"
  "    2. IB - Layer 5 intrinsically bursting cell (fig 8.19 from 2007 book)\n"
  "    3. CH - Cat primary visual cortex chattering cell (fig8.23 from 2007 book)\n"
  "    4. LTS - Rat barrel cortex Low-threshold  spiking interneuron (fig 8.25 from 2007 book)\n"
  "    5. FS - Rat visual cortex layer 5 fast-spiking interneuron (fig 8.27 from 2007 book)\n"
  "    6. TC - Cat dorsal LGN thalamocortical (TC) cell (fig 8.31 from 2007 book)\n"
  "    7. RTN - Rat reticular thalamic nucleus (RTN) cell  (fig 8.32 from 2007 book)\n"
  "ENDCOMMENT\n"
  "\n"
  ": Declare name of object and variables\n"
  "NEURON {\n"
  "  POINT_PROCESS Izhi2007a\n"
  "  RANGE C, k, vr, vt, vpeak, a, b, c, d, Iin, tauAMPA, tauNMDA, tauGABAA, tauGABAB, tauOpsin, celltype, alive, cellid, verbose\n"
  "  RANGE V, u, gAMPA, gNMDA, gGABAA, gGABAB, gOpsin, I\n"
  "  RANGE factor, eventflag, delta, t0\n"
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
  "  Vpre = 0\n"
  "  tauAMPA = 5 (ms) : Receptor time constant, AMPA\n"
  "  tauNMDA = 150 (ms) : Receptor time constant, NMDA\n"
  "  tauGABAA = 6 (ms) : Receptor time constant, GABAA\n"
  "  tauGABAB = 150 (ms) : Receptor time constant, GABAB\n"
  "  tauOpsin = 50 (ms) : Receptor time constant, opsin, from Mattis et al. (2011)\n"
  "  celltype = 1 : A flag for indicating what kind of cell it is,  used for changing the dynamics slightly (see list of cell types in initial comment).\n"
  "  alive = 1 : A flag for deciding whether or not the cell is alive -- if it's dead, acts normally except it doesn't fire spikes\n"
  "  cellid = -1 : A parameter for storing the cell ID, if required (useful for diagnostic information)\n"
  "  verbose = 0 : Whether or not to print diagnostic information to file -- WARNING, do not modify this manually -- it's set by useverbose()\n"
  "}\n"
  "\n"
  ": Variables used for internal calculations\n"
  "ASSIGNED {\n"
  "  factor : Voltage factor used for calculating the current\n"
  "  eventflag : For diagnostic information\n"
  "  V (mV) : Membrane voltage\n"
  "  u (mV) : Slow current/recovery variable\n"
  "  gAMPA : AMPA conductance\n"
  "  gNMDA : NMDA conductance\n"
  "  gGABAA : GABAA conductance\n"
  "  gGABAB : GABAB conductance\n"
  "  gOpsin : Opsin conductance\n"
  "  I : Total current\n"
  "  delta : Time step\n"
  "  t0 : Previous time  \n"
  "}\n"
  "\n"
  ": Initial conditions\n"
  "INITIAL {\n"
  "  V = vr\n"
  "  u = 0.0\n"
  "  t0 = t\n"
  "  gAMPA = 0\n"
  "  gNMDA = 0\n"
  "  gGABAA = 0\n"
  "  gGABAB = 0\n"
  "  gOpsin = 0\n"
  "  I = 0\n"
  "  delta = 0\n"
  "  net_send(0,1) : Required for the WATCH statement to be active\n"
  "}\n"
  "\n"
  "\n"
  ": Function for printing diagnostic information to a file -- usage example: cell.useverbose(2,\"logfile.txt\")\n"
  "VERBATIM\n"
  "char filename[1000]; // Allocate some memory for the filename\n"
  "ENDVERBATIM\n"
  "PROCEDURE useverbose() { : Create user-accessible function\n"
  "  VERBATIM\n"
  "  #include<stdio.h> // Basic input-output\n"
  "  verbose = (float) *getarg(1); // Set verbosity -- 0 = none, 1 = events, 2 = events + timesteps\n"
  "  strcpy(filename, gargstr(2)); // Copy input filename into memory\n"
  "  ENDVERBATIM\n"
  "}\n"
  "\n"
  ": Define neuron dynamics\n"
  "BREAKPOINT {\n"
  "  delta = t-t0 : Find time difference\n"
  "\n"
  "  : Receptor dynamics -- the correct form is gAMPA = gAMPA*exp(-delta/tauAMPA), but this is 30% slower and, in the end, not really any more physiologically realistic\n"
  "  gAMPA = gAMPA - delta*gAMPA/tauAMPA : \"Exponential\" decays -- fast excitatory (AMPA)\n"
  "  gNMDA = gNMDA - delta*gNMDA/tauNMDA : Slow excitatory (NMDA)\n"
  "  gGABAA = gGABAA - delta*gGABAA/tauGABAA : Fast inhibitory (GABA_A)\n"
  "  gGABAB = gGABAB - delta*gGABAB/tauGABAB : Slow inhibitory (GABA_B)\n"
  "  gOpsin = gOpsin - delta*gOpsin/tauOpsin : Optogenetic (opsin)\n"
  "  \n"
  "  : Calculate current\n"
  "  factor = ((V+80)/60)*((V+80)/60)\n"
  "  I = gAMPA*(V-0) + gNMDA*factor/(1+factor)*(V-0) + gGABAA*(V+70) + gGABAB*(V+90) + gOpsin*(V-0) : Treat the opsin channel like an AMPA channel\n"
  "  \n"
  "  : Calculate neuronal dynamics; -I since I = -I_{syn}, which is really what I is as I've defined it above\n"
  "  Vpre = V\n"
  "  V = V + delta*(k*(V-vr)*(V-vt) - u - I + Iin)/(C*100)  : Calculate voltage\n"
  "\n"
  "  if (Vpre<=c && V>vpeak) {V=c+1} : if just spiked, wait at least 1 timestep before increasing V>vpeak again, so V reset value takes effect; WATCH statement requires V to cross the vpeak threshod)\n"
  "\n"
  "  : Cell-type specific dynamics\n"
  "  if (celltype<5) {\n"
  "    u = u + delta*a*(b*(V-vr)-u) : Calculate recovery variable\n"
  "  }\n"
  "  else {\n"
  "     : For FS neurons, include nonlinear U(v): U(v) = 0 when v<vb ; U(v) = 0.025(v-vb) when v>=vb (d=vb=-55)\n"
  "     if (celltype==5) {\n"
  "       if (V<d) { \n"
  "        u = u + delta*a*(0-u)\n"
  "       }\n"
  "       else { \n"
  "        u = u + delta*a*((0.025*(V-d)*(V-d)*(V-d))-u)\n"
  "       }\n"
  "     }\n"
  "\n"
  "     : For TC neurons, reset b\n"
  "     if (celltype==6) {\n"
  "       if (V>-65) {b=0}\n"
  "       else {b=15}\n"
  "       u = u + delta*a*(b*(V-vr)-u) : Calculate recovery variable\n"
  "     }\n"
  "     \n"
  "     : For TRN neurons, reset b\n"
  "     if (celltype==7) {\n"
  "       if (V>-65) {b=2}\n"
  "       else {b=10}\n"
  "       u = u + delta*a*(b*(V-vr)-u) : Calculate recovery variable\n"
  "     }\n"
  "  }\n"
  "\n"
  "  t0=t : Reset last time so delta can be calculated in the next time step\n"
  "  \n"
  "  : Print diagnostic inormation to a file\n"
  "  if (verbose>1) { : Verbose turned on?\n"
  "    VERBATIM\n"
  "    FILE *outfile; // Declare file object\n"
  "    outfile=fopen(filename,\"a\"); // Open file for appending\n"
  "    fprintf(outfile,\"%8.2f   cell=%6.0f   delta=%8.2f   gAMPA=%8.2f   gNMDA=%8.2f   gGABAA=%8.2f   gGABAB=%8.2f   gOpsin=%8.2f   factor=%8.2f   I=%8.2f   V=%8.2f   u=%8.2f (timestep)\\n\",t,cellid,delta,gAMPA,gNMDA,gGABAA,gGABAB,gOpsin,factor,I,V,u);\n"
  "    fclose(outfile); // Close file\n"
  "    ENDVERBATIM\n"
  "  }\n"
  "}\n"
  "\n"
  ": Input received\n"
  "NET_RECEIVE (wAMPA, wNMDA, wGABAA, wGABAB, wOpsin) {  \n"
  "  INITIAL { wAMPA=wAMPA wNMDA=wNMDA wGABAA=wGABAA wGABAB=wGABAB wOpsin=wOpsin} : Insanely stupid but required, otherwise reset to 0, \n"
  "\n"
  "  : Check if spike occurred\n"
  "  if (flag == 1) { : Fake event from INITIAL block\n"
  "    if (celltype < 4 || celltype == 5 || celltype == 7) { : default\n"
  "      WATCH (V>vpeak) 2 : Check if threshold has been crossed, and if so, set flag=2\n"
  "    }\n"
  "    else if (celltype == 4) { : LTS cell\n"
  "      WATCH (V>(vpeak-0.1*u)) 2 : Check if threshold has been crossed, and if so, set flag=2     \n"
  "    }\n"
  "    else if (celltype == 6) { : TC cell\n"
  "      WATCH (V>(vpeak+0.1*u)) 2 : Check if threshold has been crossed, and if so, set flag=2     \n"
  "    }\n"
  "  } \n"
  "  \n"
  "  : Event created by WATCH statement -- i.e. threshold crossed\n"
  "  else if (flag == 2) { \n"
  "    if (alive) {net_event(t)} : Send spike event if the cell is alive\n"
  "\n"
  "    : For RS, IB and CH neurons, and RTN\n"
  "    if (celltype < 4 || celltype == 7) {\n"
  "      V = c : Reset voltage\n"
  "      u = u+d : Reset recovery variable\n"
  "    }\n"
  "    : For LTS neurons\n"
  "    else if (celltype == 4) {\n"
  "      V = c+0.04*u : Reset voltage\n"
  "      if ((u+d)<670) {u=u+d} : Reset recovery variable\n"
  "      else {u=670} \n"
  "     }  \n"
  "    : For FS neurons (only update v)\n"
  "    else if (celltype == 5) {\n"
  "      V = c : Reset voltage\n"
  "     }  \n"
  "    : For TC neurons (only update v)\n"
  "    else if (celltype == 6) {\n"
  "      V = c-0.1*u : Reset voltage\n"
  "      u = u+d : Reset recovery variable\n"
  "     }  \n"
  "\n"
  "    gAMPA = 0 : Reset conductances -- not mentioned in Izhikevich's paper but necessary to stop things from exploding!\n"
  "    gNMDA = 0\n"
  "    gGABAA = 0\n"
  "    gGABAB = 0\n"
  "    gOpsin = 0\n"
  "  } \n"
  "  \n"
  "  : Actual input, calculate receptor dynamics\n"
  "  else {\n"
  "    gAMPA = gAMPA + wAMPA\n"
  "    gNMDA = gNMDA + wNMDA\n"
  "    gGABAA = gGABAA + wGABAA\n"
  "    gGABAB = gGABAB + wGABAB\n"
  "    gOpsin = gOpsin + wOpsin\n"
  "  }\n"
  "  \n"
  "  : Print diagnostic information to a file\n"
  "  if (verbose>0) { : Verbose turned on?\n"
  "    eventflag = flag\n"
  "    VERBATIM\n"
  "    FILE *outfile; // Declare file object\n"
  "//if(cellid>=0 && cellid < 300) {\n"
  "    outfile=fopen(filename,\"a\"); // Open file for appending\n"
  "    fprintf(outfile,\"t=%8.2f   cell=%6.0f   flag=%1.0f   gAMPA=%8.2f   gNMDA=%8.2f   gGABAA=%8.2f   gGABAB=%8.2f   gOpsin=%8.2f   V=%8.2f   u=%8.2f (event)\\n\",t, cellid,eventflag,gAMPA,gNMDA,gGABAA,gGABAB,gOpsin,V,u);\n"
  "    fclose(outfile); // Close file\n"
  "//}\n"
  "    ENDVERBATIM\n"
  "  }\n"
  "  \n"
  "  \n"
  "}\n"
  ;
#endif
