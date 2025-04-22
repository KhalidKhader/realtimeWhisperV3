from jiwer import wer

reference = """
pécialités en cardiologie 
cardiologie  
cardiologie interventionnelle  
électrophysiologie  
cardiologie pédiatrique  
spécialiste des maladies cardiaques congénitales  
chirurgie cardiothoracique  
chirurgie vasculaire  
cardiologie préventive  
spécialiste de linsuffisance cardiaque  
spécialiste de lhypertension  
cardiologie de transplantation  
cardiologie du sport  
insuffisance cardiaque avancée et transplantation cardiaque  
cardiologie nucléaire  
spécialiste en échocardiographie  
spécialiste en imagerie cardiaque  

aladies et affections élargies
athérosclérose  
angine de poitrine stable instable  
infarctus du myocarde stemi nstemi  
maladie coronarienne cad  
maladie cardiaque ischémique  
arythmie fibrillation auriculaire tachycardie ventriculaire etc  
bradycardie  
tachycardie  
insuffisance cardiaque systolique diastolique  
insuffisance cardiaque congestive chf  
cardiomyopathie dilatée hypertrophique restrictive  
péricardite  
endocardite  
myocardite  
maladie rhumatismale du cœur  
prolapsus de la valve mitrale  
sténose aortique  
régurgitation mitrale  
régurgitation tricuspidienne  
hypertension pulmonaire  
thrombose veineuse profonde tvp  
embolie pulmonaire ep  
maladie artérielle périphérique map  
cardiomyopathie hypertrophique obstructive cmho  
syndrome du qt long  
syndrome de brugada  
syndrome de wolffparkinsonwhite  
tétralogie de fallot  
canal artériel persistant cap  
communication interventriculaire civ  
communication interauriculaire cia  
coarctation de laorte  
transposition des gros vaisseaux  
syndrome deisenmenger  
syndrome de marfan  
cardiomyopathie de takotsubo syndrome du cœur brisé  
tamponnade cardiaque  
choc cardiogénique  
arrêt cardiaque soudain  
syncope  
hypotension orthostatique  
syncope vasovagale  
syndrome de tachycardie orthostatique posturale stomp  
cardiomyopathie dilatée  
cardiomyopathie restrictive  
endocardite infectieuse  
cardiomyopathie non compactée  
anévrisme de lartère coronaire  
dissection de lartère coronaire  
spasme coronaire  
angine variante  
angine de prinzmetal  
cardiomyopathie induite par le stress  
sarcoïdose cardiaque  
amylose  
amylose cardiaque  
fibrome cardiaque  
lipome cardiaque  
rhabdomyome cardiaque  
hémangiome cardiaque  
myxome cardiaque  
sarcome cardiaque  
lymphome cardiaque  
métastase cardiaque  
thrombus cardiaque  
embolie cardiaque  
rupture cardiaque  
rupture de la paroi libre ventriculaire  
rupture septale  
rupture du muscle papillaire  

édicaments élargis
aspirine  
clopidogrel  
ticagrelor  
prasugrel  
warfarine  
rivaroxaban  
apixaban  
edoxaban  
dabigatran  
bêtabloquants métropolol atenolol propranolol  
inhibiteurs calciques amlodipine diltiazem vérapamil  
inhibiteurs de lenzyme de conversion iec lisinopril énalapril ramipril  
antagonistes des récepteurs de langiotensine ii ara ii losartan valsartan irbesartan  
diurétiques furosémide hydrochlorothiazide spironolactone  
statines atorvastatine rosuvastatine simvastatine  
nitrates nitroglycérine isosorbide dinitrate  
digitaline  
amiodarone  
sotalol  
flecainide  
propafénone  
ivabradine  
sacubitril/valsartan entresto  
éplérénone  
dobutamine  
milrinone  
héparine  
héparine de bas poids moléculaire hbpm  
fondaparinux  
adénosine  
atropine  
épinéphrine  
vasopressine  
nésiritide  
lévosiémendan  
ranolazine  
trimétazidine  
nicorandil  
ivabradine  
molsidomine  
hydralazine  
isosorbide mononitrate  
carvédilol  
bisoprolol  
nébivolol  
aliskiren  
époprosténol  
tréprostinil  
bosentan  
macitentan  
riociguat  

ermes anatomiques élargis
cœur  
oreillette gauche  
oreillette droite  
ventricule gauche  
ventricule droit  
aorte  
artère pulmonaire  
veines pulmonaires  
veine cave supérieure  
veine cave inférieure  
artères coronaires interventriculaire antérieure gauche circonflexe artère coronaire droite  
valve mitrale  
valve tricuspide  
valve aortique  
valve pulmonaire  
péricarde  
endocarde  
myocarde  
épicarde  
sinus coronaire  
faisceau de his  
fibres de purkinje  
nœud sinoauriculaire sa  
nœud auriculoventriculaire av  
septum interventriculaire  
appendice auriculaire  
muscles papillaires  
cordes tendineuses  
anneau fibreux  
squelette fibreux du cœur  
bande modératrice  
crista terminalis  
fosse ovale  
ligament artériel  
sinus de valsalva  
ostium coronaire  
venae cordis minimae  

rocédures médicales élargies
électrocardiogramme ecg/ecg  
moniteur holter  
moniteur dévénements  
test deffort exercice pharmacologique  
échocardiogramme transthoracique transœsophagien  
cathétérisme cardiaque  
angiographie coronaire  
intervention coronarienne percutanée icp  
pose de stent  
angioplastie par ballonnet  
athérectomie  
cardioversion électrique chimique  
implantation de stimulateur cardiaque  
défibrillateur cardioverteur implantable dci  
ablation cardiaque  
procédure de maze  
pontage aortocoronarien cabg  
transplantation cardiaque  
dispositif dassistance ventriculaire dav  
pompe à ballonnet intraaortique pbia  
réanimation cardiopulmonaire rcp  
défibrillation  
thrombolyse  
angiographie par tomodensitométrie tdm  
irm cardiaque  
test deffort nucléaire  
tep cardiaque  
biopsie endomyocardique  
cathétérisme cardiaque gauche  
cathétérisme cardiaque droit  
cathétérisme de swanganz  
échographie intravasculaire ivus  
tomographie par cohérence optique oct  
réserve de flux fractionnaire ffr  
rapport sans onde instantané ifr  
imagerie par résonance magnétique cardiaque irm cardiaque  
tomodensitométrie cardiaque tdm cardiaque  
remplacement valvulaire aortique par cathétérisme tavi  
procédure mitraclip  
fermeture de lappendice auriculaire gauche faag  
isolation des veines pulmonaires ivp  
cartographie épica

diale  
cartographie endocardiale  
thérapie de resynchronisation cardiaque trc  

ermes cliniques élargis
systole  
diastole  
volume systolique  
débit cardiaque  
fraction déjection  
précharge  
postcharge  
fréquence cardiaque  
pression artérielle systolique diastolique  
pression pulsée  
pression artérielle moyenne pam  
hypertension primaire secondaire  
hypotension  
dyspnée  
orthopnée  
dyspnée paroxystique nocturne dpn  
œdème périférique pulmonaire  
distension jugulaire veineuse djv  
souffle systolique diastolique  
rythme galop s3 s4  
palpitations  
cyanose  
hippocratisme  
douleur thoracique angineuse non angineuse  
claudication  
syncope  
présyncope  
étourdissements  
fatigue  
faiblesse  
vertige  
intolérance orthostatique  
pulsus paradoxus  
pulsus alternans  
signe de kussmaul  
triade de beck  
triade de virchow  
nodosités dosler  
lésions de janeway  
taches de roth  
hémorragies en fuseau  
pouls en marteau deau  
pouls de corrigan  
temps de recoloration capillaire  
pression veineuse centrale pvc  
pression capillaire pulmonaire pcp  
index cardiaque ic  
résistance vasculaire systémique rvs  
résistance vasculaire pulmonaire rvp  
saturation en oxygène spo2  
saturation veineuse mixte en oxygène svo2
"""
hypothesis = """
anadian french script spécialité en cardiologie cardiologie cardiologie interventionnelle
 électrophysiologie cardiologie pédiatrique spécialiste des maladies cardiaques congénitales chirurgie cardiotoracique
 chirurgie vasculaire cardiologie préventive spécialiste de linsuffisance cardiaque spécialiste de lhypertension
 cardiologie de transplantation cardiologie du sport insuffisance cardiaque avancée et transplantation cardiaque cardiologie
 nucléaire spécialiste en éco cardiographie spécialiste en imagerie cardiaque maladie et affections élargie
 athérosclérose angine de poitrine stable instable infarctus du myocarde stémie en stémie maladie
 cad maladie cardiaque ischémique arythmie fibrillation auriculaire tachycardie ventriculaire etc
 bradycardie tachycardie insuffisance cardiaque systolique diastolique insuffisance cardiaque congestive
 chf cardiomyopathie dilatée hypertrophique restrictive péricardite endocardite
 myocardite maladie rhumatismale du coeur prolapsus de la valve mitrale sténose aortique régurgitation
 mitrale régurgitation trituspidienne hypertension pulmonaire thrombose veineuse profonde tvp
 polypulmonaire ap maladie artérielle périphérique map cardiomyopathie hypertrophique obstructive cmho
 syndrome du cutélon syndrome de brugada syndrome de wolf-parkinson-white tétralogie de fallot canal artériel
 cap communication interventriculaire civ communication inter auriculaire cia coarctation de la haute
 transposition des gros vaisseaux syndrome des en manger syndrome de marfan cardiomyopathie de takotsubo
 syndrome du cœur brisé tamponade cardiaque choc cardiogénique arrêt cardiaque soudain syncope
 hypotension orthostatique syncope vasovagale syndrome de tachycardie orthostatique posturale stomp cardiomyopathie
 dilatée cardiomyopathy restrictive endocardite infectieuse cardiomyopathie non compactée anévrisme de lart
 coroner dissection de lartère coronaire spasme coronaire angine variante angine de prinsmuthal
 cardiomyopathie induite par le stress sarcoïdose cardiaque amylose amylose cardiaque fibros
 cardiaque lipome cardiaque rhabdomyome cardiaque hemangium cardiaque myxome cardiaque
 sarcombe cardiaque lymphome cardiaque métastase cardiaque thrombus cardiaque embolie cardiaque
 rupture cardiaque rupture de la paroi libre ventriculaire rupture septale rupture du muscle papillaire
 médicaments élargis aspirine clopidogrel ticagrelor prasugrel
 farine rivaroxaban apixaban et doxaban dabigatran bétabloquins métropole
 athénolol propranolol inhibiteur calcique enlodipine diltiazem verapamil inhibiteur de lenzyme de conversion
 lisinopril enalapril ramipril antagoniste des récepteurs de langiotensine 2 ara2 losartin
 valsartin herbeux artin diurétique furosémide hydrochlorothiazide spironolactone statine atorvastatine rosuvace
 sévastatine sévastatine nitrate nitroglucérine isosorbite dinitrate digitaline amiodarone
 sotalol flequenide propaphenone ivabradine sacubitril et valsartan entresto
 eplérénone dobutamine milrinone éparine éparine de bas poids moléculaire hbpm
 fond daparinux adénosine atropine épinephrine vasopressine nésit
 les vous y mandant ranolasine trimétasidine nicorandile ivabradine
 molcidomine hydralazine isosorbide mononitrate carvedilol bisoprolol
 nébivolol alischirène époprosténol préprostinil bosantin macitant
 ryosiga termes anatomiques élargis coeur oreillette gauche oreillette droite
 ventricule gauche ventricule droit aortes artère pulmonaire veine pulmonaire veine cave
 supérieure veine cave inférieure artère coronaire interventriculaire antérieure gauche circonflexe artère coronaire droite
 valve mitral valve tricuspide valve aortique valve pulmonaire péricarde
 endocarde myocarde épicarde sinus coronaire faisceau de his fibres de purquin
 le sino auriculaire est ça ne auriculoventriculaire av sept hommes interventriculaire appendice
 muscles papillaires cordes tendineuses anneaux fibrous squelettes fibrous du coeur bande modélique
 crista terminali fosse ovale ligament artériel sinus de valsalva osteoporosis
 coroner venait cordy minimaille procédure médicale élargie électrocardiogramme le cg et cg
 moniteur olt moniteur dévénements test defforts et exercices pharmacologiques échocardiogramme transthoracique
 transœsophagien cathétérisme cardiaque angiographie coronaire intervention coronarienne percutanée icp
 pose de stand angioplastie par ballonnet athérectomie cardioversion électrique chimique
 implantation de stimulateur cardiaque défibriateur cardioverteur implantable dci ablation cardiaque procédure de
 mase pontage aorto-coronarien câbles transplantation cardiaque dispositif dassistance ventriculaire dav
 pompe à ballonnets intra-aortiques pbia réanimation cardio pulmonaire rcp défibrillation thrombolite
 angiographie par thomodensitométrie tdm irm cardiaque test defforts nucléaires tep cardiaque
 biopsie endoméocardique cathétérisme cardiaque gauche cathétérisme cardiaque droit cathétérisme de soin
 échographie intravasculaire ivus tomographie par cohérence optique oct réserve de flux fractionnaire
 rapport sans ondes instantanées ifr imagerie par résonance magnétique cardiaque irm cardiaque thomodensitométrie
 cardiaque tdm cardiaque remplacement valvulaire aortique par cathétérisme tavi procédure mitraclip fermeture
 de lappendice auriculaire gauche fag isolation des veines pulmonaires ivp cartographie epica rodial
  cartographie endocardiale thérapie de resynchronisation cardiaque trc termes cliniques élargies systole
 diastole volume systolique débit cardiaque fraction déjection précharge
 post-charge fréquence cardiaque pression artérielle systolique diastolique pression pulsée pression
 artérielle moyenne pam hypertension primaire secondaire hypotension dyspnée orthopnée
 dyspnée paroxystique nocturne dpn œdème périphérique pulmonaire distension jugulaire veineuse dgv
 souffle systolique diastolique rythme galop s3 s4 palpitations cyanose
 douleur thoracique angineuse non angineuse claudication syncope
 étourdissement fatigue faiblesse vertige intolérance orthostatique
 pulsus paradoxus pulsus alternant signe de kussmol triade de beck triade de virchow
 nodosité dosselet lésions de janouet taches de rosse hémorragie en fuseau pou en marteau deau
 de corrigan temps de recoloration capillaire pression veineuse centrale pvc pression capillaire pulmonaire pcp
 index cardiaque ic résistance vasculaire systémique arvs résistance vasculaire pulmonaire arvp
 saturation en oxygène uspo2 saturation veineuse mixte en oxygène svo2
"""

error = (wer(reference , hypothesis))
print(error)



