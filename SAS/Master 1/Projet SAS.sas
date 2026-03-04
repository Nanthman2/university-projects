options orientation=landscape;
title "projet SAS 2024 : partie A";
ods pdf file='/home/u64074511/sasuser.v94/sorties_projetSAS2024.pdf' ;
libname projet "/home/u64074511/my_shared_file_links/celinebugli/LSTAT2020/projetSAS2024";
data _null_;
	set remote_bis;
	file '/home/u64074511/sasuser.v94/remote_bis.txt';
	put Employee_ID Age date gender Job_Role Industry Years_of_Experience Work_Location
		Hours_Worked_Per_Week Number_of_Virtual_Meetings Work_Life_Balance_Rating 
		Stress_Level productivity_change Social_Isolation_Rating 
		Satisfaction_with_Remote_Work Company_Support_for_Remote_Work 
		Physical_Activity Sleep_Quality Region;
run;






data projet;
   set projet.remote;
run;
proc contents data=projet;
run;
proc freq data=projet;
    tables region / nopercent;
run;
proc format;
  value niveau
    1 = "très faible"
    2 = "faible"
    3 = "modéré"
    4 = "élevé"
    5 = "très élevé";
run;
data work;
  set projet.remote;
  format work_life_balance_rating niveau.;
  if 18 < (age - years_of_experience) 
  then check = 1;
  else check = 0;
run;
proc freq data=work;
    tables work_life_balance_rating*region / nofreq nopercent;
run;
proc freq data=work;
  tables check;
run;
data work; 
    set work; 
    if years_of_experience < (age - 18) then experience_clean = years_of_experience;
    else experience_clean = .;
run;
proc means data=work n nmiss mean median min max var maxdec=2;
    var years_of_experience experience_clean ;
run;
proc sort data=work;
    by Region;
run;
proc means data=work n nmiss mean median min max var maxdec=2;
    var years_of_experience experience_clean;
    by Region;
run;
proc format;
    value form_hours
        low-<30 = '<30h'
        30-<40 = '30-40h'
        40-<50 = '40-50h'
        50-high = '>50h';
run;
data remote;
   set projet.remote;
run;
proc print data=remote (obs=10);
	format Hours_worked_per_week form_hours.;
	where Region = 'Europe';
	var Employee_ID Region Hours_worked_Per_Week;
run;
data remote_graph;
    set remote;
    where Hours_worked_per_week < 40;
run;
title "";
/* -------------------------------------------------------------------
   Code généré par une tâche SAS

   Généré le : dimanche 15 décembre 2024 à 17:11:45
   Par tâche : Boîte à moustaches 3

   Données d'entrée : SASApp:WORK.REMOTE_GRAPH
   Serveur :  SASApp
   ------------------------------------------------------------------- */

%_eg_conditional_dropds(WORK.SORTTempTableSorted);
/* -------------------------------------------------------------------
   Trier la table WORK.REMOTE_GRAPH
   ------------------------------------------------------------------- */
PROC SORT
	DATA=WORK.REMOTE_GRAPH(KEEP=Job_Role Hours_Worked_Per_Week Industry)
	OUT=WORK.SORTTempTableSorted
	;
	BY Industry;
RUN;
SYMBOL1 	INTERPOL=BOXF	CV=BLUE
	VALUE=CIRCLE
	HEIGHT=1
	MODE=EXCLUDE
;
Axis1
	STYLE=1
	WIDTH=1
	MINOR=NONE

;
Axis2
	STYLE=1
	WIDTH=1
	MINOR=NONE
	LABEL=(HEIGHT=8pt )

	VALUE=(HEIGHT=8pt )
;
TITLE;
TITLE1 "Boîte à moustaches";
FOOTNOTE;
FOOTNOTE1 "projet SAS 2024";
PROC GPLOT DATA=WORK.SORTTempTableSorted
 NOCACHE ;
	PLOT Hours_Worked_Per_Week * Job_Role/
	VAXIS=AXIS1

	HAXIS=AXIS2

;
	BY Industry;
/* -------------------------------------------------------------------
   Fin du code de la tâche
   ------------------------------------------------------------------- */
RUN; QUIT;
%_eg_conditional_dropds(WORK.SORTTempTableSorted);
TITLE; FOOTNOTE;
GOPTIONS RESET = SYMBOL;

ods pdf close;
data remote_bis;
    set projet.remote_raw2;
    date_charactere = put(date, 8.);
    annee = substr(date_charactere, 5, 4);
    mois = substr(date_charactere, 3, 2);
    jour = substr(date_charactere, 1, 2);
	date_SAS = mdy(input(mois, 2.), input(jour, 2.), input(annee, 4.));
retain ID 0;
    ID+1;
    Employee_ID = cat('EMP', put(ID, z4.));
keep Employee_ID Age date gender Job_Role Industry Years_of_Experience Work_Location Hours_Worked_Per_Week Number_of_Virtual_Meetings Work_Life_Balance_Rating Stress_Level productivity_change Social_Isolation_Rating Satisfaction_with_Remote_Work Company_Support_for_Remote_Work Physical_Activity Sleep_Quality Region;
run;
proc compare base=projet.remote compare=remote_bis;
run;

