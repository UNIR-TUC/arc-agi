ATENCIÓN ES TODO LO QUE NECESITAS
AshishVaswani∗ NoamShazeer* NikiParmar* JakobUszkoreit
GoogleBrain GoogleBrain GoogleResearch GoogleResearch
avaswani@google.com noam@google.com nikip@google.com usz@google.com
LlionJones AidanN.Gomez† ŁukaszKaiser
GoogleResearch UniversityofToronto GoogleBrain
llion@google.com aidan@cs.toronto.edu lukaszkaiser@google.com
IlliaPolosukhin‡
illia.polosukhin@gmail.com
ABSTRACT
Losmodelos detransduccióndesecuencias dominantessebasanen redesneuronalescomplejas,
recurrentes o convolucionales, que incluyen un codificador y un decodificador. Los modelos de
mejorrendimientotambiénconectanelcodificadoryeldecodificadormedianteunmecanismode
atención. Proponemosunanuevaarquitecturaderedsimple,elTransformer,basadaúnicamenteen
mecanismosdeatención,prescindiendoporcompletodelarecurrenciaylasconvoluciones. Los
experimentosendostareasdetraducciónautomáticamuestranqueestosmodelossonsuperiores
en calidad, al mismo tiempo que son más paralelizables y requieren mucho menos tiempo para
entrenarlos. Nuestromodelologra28,4BLEUenlatareadetraduccióndelinglésalalemándel
WMT2014,mejorandolosmejoresresultadosexistentes,incluidoslosconjuntos,enmásde2BLEU.
EnlatareadetraduccióndelinglésalfrancésdeWMT2014,nuestromodeloestableceunanueva
puntuaciónBLEUdeúltimageneracióndeunsolomodelode41,8despuésdeunentrenamiento
durante3,5díasenochoGPU,unapequeñafraccióndeloscostosdeentrenamientodelosmejores.
modelosdelaliteratura. MostramosqueTransformersegeneralizabienaotrastareasalaplicarlo
conéxitoalanálisisdedistritoselectoraleseningléscondatosdeentrenamientograndesylimitados.
∗Igualcontribución.Elordendelistadoesaleatorio.JakobpropusoreemplazarlosRNNconautoatenciónycomenzóelesfuerzo
deevaluarestaidea. Ashish,conIllia,diseñóeimplementólosprimerosmodelosdeTransformeryhaparticipadodemanera
crucialentodoslosaspectosdeestetrabajo.Noampropusolaatencióndeproductoescalado,laatencióndemúltiplescabezasy
larepresentacióndeposiciónsinparámetrosyseconvirtióenlaotrapersonainvolucradaencasitodoslosdetalles.Nikidiseñó,
implementó,ajustóyevaluóinnumerablesvariantesdemodelosennuestrocódigobaseoriginalytensor2tensor. Lliontambién
experimentóconnuevasvariantesdemodelos,fueresponsabledenuestrabasedecódigoinicialydeinferenciasyvisualizaciones
eficientes.LukaszyAidanpasaroninnumerablesdíasdiseñandovariasparteseimplementandotensor2tensor,reemplazandonuestro
códigobaseanterior,mejorandoenormementelosresultadosyacelerandoenormementenuestrainvestigación.
†TrabajorealizadomientrasestabaenGoogleBrain.
‡TrabajorealizadomientrasestabaenGoogleResearch.

1 Introducción
Las redes neuronales recurrentes, la memoria a largo plazo [13] y las redes neuronales recurrentes cerradas [7]
enparticular, sehanestablecidofirmementecomoenfoquesdeúltimageneraciónenelmodeladodesecuenciasy
problemasdetransducción, comoelmodeladodellenguajeylatraducciónautomática[35,2,5]. Desdeentonces,
numerososesfuerzoshanseguidoampliandoloslímitesdelosmodelosdelenguajerecurrentesylasarquitecturasde
codificador-decodificador[38,24,15]. Losmodelosrecurrentessuelenfactorizarelcálculosegúnlasposicionesde
lossímbolosdelassecuenciasdeentradaysalida. Alalinearlasposicionesconlospasoseneltiempodecálculo,
generanunasecuenciadeestadosocultosh ,enfuncióndelestadoocultoanteriorh ylaentradaparalaposiciónt.
t t−1
Estanaturalezainherentementesecuencialimpidelaparalelizacióndentrodelosejemplosdeentrenamiento,loque
sevuelvecríticoensecuenciasdemayorlongitud,yaquelaslimitacionesdememorialimitanelprocesamientopor
lotesentreejemplos. Trabajosrecienteshanlogradomejorassignificativasenlaeficienciacomputacionalatravésde
trucosdefactorización[21]ycálculocondicional[32],altiempoquemejoranelrendimientodelmodeloenelcaso
deesteúltimo. Sinembargo,persistelalimitaciónfundamentaldelcálculosecuencial. Losmecanismosdeatención
sehanconvertidoenunaparteintegraldelmodeladodesecuenciasconvincentesylosmodelosdetransducciónen
diversastareas,permitiendomodelardependenciassintenerencuentasudistanciaenlassecuenciasdeentradaosalida
[2,19]. Sinembargo,entodosloscasos,exceptoenunospocos[27],dichosmecanismosdeatenciónseutilizanjunto
conunaredrecurrente. EnestetrabajoproponemoselTransformer,unaarquitecturamodeloqueevitalarecurrenciay,
encambio,sebasacompletamenteenunmecanismodeatenciónparadibujardependenciasglobalesentreentraday
salida. ElTransformerpermiteunaparalelizaciónsignificativamentemayorypuedealcanzarunnuevoestadodelarte
encalidaddetraduccióndespuésdehabersidoentrenadodurantetansolodocehorasenochoGPUP100.
2 Background
El objetivo de reducir el cálculo secuencial también forma la base de Extended Neural GPU [16], ByteNet [18] y
ConvS2S[9],todosloscualesutilizanredesneuronalesconvolucionalescomobloquedeconstrucciónbásico,calculando
representaciones ocultas en paralelo para todas las entradas y salidas. posiciones de salida. En estos modelos, el
númerodeoperacionesnecesariaspararelacionarseñalesdedosposicionesdeentradaosalidaarbitrariascreceenla
distanciaentreposiciones,linealmenteparaConvS2SylogarítmicamenteparaByteNet. Estohacequeseamásdifícil
aprenderlasdependenciasentreposicionesdistantes[12]. EnelTransformador,estosereduceaunnúmeroconstante
deoperaciones,aunqueacostadeunaresoluciónefectivareducidadebidoalpromediodeposicionesponderadaspor
atención,unefectoquecontrarrestamosconAtencióndemúltiplescabezalescomosedescribeenlasección3.2.
Laautoatención,avecesllamadaintraatención,esunmecanismodeatenciónquerelacionadiferentesposicionesde
unasolasecuenciaparacalcularunarepresentacióndelasecuencia. Laatenciónpersonalsehautilizadoconéxitoen
unavariedaddetareasqueincluyencomprensiónlectora,resúmenesabstractivos,vinculacióntextualyaprendizajede
representacionesdeoracionesindependientesdelatarea[4,27,28,22].
Lasredesdememoriadeunextremoaotrosebasanenunmecanismodeatenciónrecurrenteenlugardeunarecurrencia
alineada con secuencias y se ha demostrado que funcionan bien en tareas de modelado de lenguaje y respuesta a
preguntasenlenguajesimple[34].
Sinembargo,hastadondesabemos,Transformereselprimermodelodetransducciónquesebasacompletamenteenla
autoatenciónparacalcularrepresentacionesdesuentradaysalidasinutilizarRNNalineadosensecuenciaoconvolución.
Enlassiguientessecciones,describiremoselTransformer,motivaremoslaatenciónpersonalydiscutiremossusventajas
sobremodeloscomo[17,18]y[9].
3 Arquitecturamodelo
Lamayoríadelosmodeloscompetitivosdetransduccióndesecuenciasneuronalestienenunaestructuracodificador-
decodificador [5, 2, 35]. Aquí, el codificador asigna una secuencia de entrada de representaciones de símbolos
(x ,...,x ) a una secuencia de representaciones continuas z = (z ,...,z ). Dado z, el decodificador genera una
1 n 1 n
secuenciadesalida(y ,...,y )desímbolos,unelementoalavez. Encadapaso,elmodeloesautorregresivo[10],
1 m
consumiendolossímbolosgeneradospreviamentecomoentradaadicionalalgenerarelsiguiente.
ElTransformersigueestaarquitecturageneralutilizandoautoatenciónapiladaycapaspuntualesycompletamente
conectadastantoparaelcodificadorcomoparaeldecodificador,comosemuestraenlasmitadesizquierdayderechade
laFigura1,respectivamente.
2

Figure1: ElTransformador-modeloarquitectura.
3.1 Pilasdecodificadoresydecodificadores
Codificador: ElcodificadorestácompuestoporunapiladeN = 6capasidénticas. Cadacapatienedossubcapas.
Elprimeroesunmecanismodeautoatencióndemúltiplescabezalesyelsegundoesunareddealimentacióndirecta
simple,posicionalycompletamenteconectada. Empleamosunaconexiónresidual[11]alrededordecadaunadelas
dos subcapas, seguida de una normalización de capa [1]. Es decir, la salida de cada subcapa es LayerNorm(x+
Sublayer(x)),dondeSublayer(x)eslafunciónimplementadaporlapropiasubcapa. Parafacilitarestasconexiones
residuales, todas las subcapas del modelo, así como las capas de incrustación, producen resultados de dimensión
d =512.
model
Decodificador: El decodificador también se compone de una pila de N = 6 capas idénticas. Además de las dos
subcapasencadacapadelcodificador,eldecodificadorinsertaunatercerasubcapa,querealizaatencióndemúltiples
cabezalessobrelasalidadelapiladelcodificador. Demanerasimilaralcodificador,empleamosconexionesresiduales
alrededor de cada una de las subcapas, seguidas de la normalización de capas. También modificamos la subcapa
deautoatenciónenlapiladedecodificadoresparaevitarquelasposicionesatiendanaposicionesposteriores. Este
enmascaramiento,combinadoconelhechodequelasincorporacionesdesalidaestáncompensadasenunaposición,
garantizaquelasprediccionesparalaposiciónipuedandependersólodelassalidasconocidasenposicionesmenores
quei.
3.2 Atención
Unafuncióndeatenciónsepuededescribircomoelmapeodeunaconsultayunconjuntodeparesclave-valorauna
salida,dondelaconsulta,lasclaves,losvaloresylasalidasontodosvectores. Lasalidasecalculacomounasuma
3

Figure2: (izquierda)Atencióndeproductoescalado. (derecha)Laatenciónmulticabezalconstadevariascapasde
atenciónqueseejecutanenparalelo.
ponderadadelosvalores,dondeelpesoasignadoacadavalorsecalculamedianteunafuncióndecompatibilidaddela
consultaconlaclavecorrespondiente.
3.2.1 Atencióndelproductoescalado
Llamamosnuestraatenciónparticular"Atencióndeproductoescalado"(figura2). Laentradaconstadeconsultasy
clavesdedimensiónd
k
yvalore√sdedimensiónd
v
. Calculamoslosproductosescalaresdelaconsultacontodaslas
claves,dividimoscadaunapor d yaplicamosunafunciónsoftmaxparaobtenerlospesosdelosvalores. Enla
k
práctica,calculamoslafuncióndeatenciónenunconjuntodeconsultassimultáneamente,empaquetadasenunamatriz
Q. LasclavesylosvalorestambiénseempaquetanenmatricesK yV. Calculamoslamatrizderesultadoscomo:
QKT
Atencion(Q,K,V)=sofmax( √ )V (1)
d
k
Lasdosfuncionesdeatenciónmásutilizadassonlaatenciónaditiva[2]ylaatencióndeproductoescalar(multiplicativa).
Laatencióndelproductoescalaresidénticaanuestroalgoritmo,exceptoporelfactordeescalade √1 . Laatención
dk
aditivacalculalafuncióndecompatibilidadutilizandounareddeavanceconunaúnicacapaoculta. Sibienlosdosson
similaresencomplejidadteórica,laatencióndelproductopuntoesmuchomásrápidayeficienteenelespacioenla
práctica,yaquesepuedeimplementarutilizandouncódigodemultiplicacióndematricesaltamenteoptimizado.
Mientrasqueparavalorespequeñosded losdosmecanismosfuncionandemanerasimilar,laatenciónaditivasupera
k
laatencióndelproductoescalarsinescalarparavaloresmásgrandesded [3]. Sospechamosqueparavaloresgrandes
k
ded ,losproductosescalarescrecenenmagnitud,empujandolafunciónsoftmaxaregionesdondetienegradientes
k
extremadamentepequeños4. Paracontrarrestaresteefecto,escalamoslosproductosescalaresen √1 .
dk
3.2.2 Atencióndemúltiplescabezales
Enlugarderealizarunaúnicafuncióndeatenciónconclaves,valoresyconsultasd −dimensional,encontramos
model
beneficiosoproyectarlinealmentelasconsultas,clavesyvaloreshvecescondiferentesproyeccioneslinealesaprendidas
adimensionesd ,d yd ,respectivamente. Encadaunadeestasversionesproyectadasdeconsultas,clavesyvalores,
k k v
realizamoslafuncióndeatenciónenparalelo,generandovaloresdesalidad −dimensional. Estosseconcatenany
v
unavezmásseproyectan,dandocomoresultadolosvaloresfinales,comosemuestraenlaFigura2.
Laatencióndemúltiplescabezalespermitequeelmodeloatiendademaneraconjuntainformacióndediferentessub
espaciosderepresentaciónendiferentesposiciones. Conunasolacabezadeatención,elpromedioinhibeesto.
4Parailustrarporquélosproductosescalaresaumentandetamaño,supongamosqueloscomponentesdeqyksonvariables
aleatoriasindependientesconmedia0yvarianza1.Entoncessuproductoescalar,q·k=
(cid:80)dk
q k ,tienemedia0yvarianzadk.
i=1 i i
4

MultiHead(Q,K,V)=Concat(head1,...,head )WO
h
wherehead =Attention(QWQ,KWK,KWV)
i i i i
Donde las proyecciones son matrices de parámetros WQϵRdmodel×dk,WKϵRdmodel×dk,WVϵRdmodel×dv y
i i i
WOϵRhdv×dmodel.
Enestetrabajoempleamosh=8capasdeatenciónparalelas,ocabezas. Paracadaunodeestosusamosd =d =
k v
d odel/h = 64. Debido a la dimensión reducida de cada cabeza, el costo computacional total es similar al de la
m
atencióndeunasolacabezacondimensionalidadcompleta.
3.2.3 AplicacionesdelaAtenciónennuestroModelo
ElTransformerutilizalaatencióndemúltiplescabezalesdetresmanerasdiferentes:
• Enlascapasde"atencióncodificador-decodificador",lasconsultasprovienendelacapadecodificadoraanterior
ylasclavesyvaloresdememoriaprovienendelasalidadelcodificador. Estopermitequecadaposiciónenel
decodificadoratiendatodaslasposicionesenlasecuenciadeentrada. Estoimitalosmecanismostípicosde
atencióncodificador-decodificadorenmodelossecuenciaasecuenciacomo[38,2,9].
• El codificador contiene capas de autoatención. En una capa de autoatención, todas las claves, valores y
consultasprovienendelmismolugar,enestecaso,lasalidadelacapaanteriorenelcodificador. Cadaposición
enelcodificadorpuedeatenderatodaslasposicionesenlacapaanteriordelcodificador.
• Demanerasimilar,lascapasdeautoatencióneneldecodificadorpermitenquecadaposicióneneldecodificador
atiendatodaslasposicioneseneldecodificadorhastaesaposicióninclusive. Necesitamosevitarelflujode
informaciónhacialaizquierdaeneldecodificadorparapreservarlapropiedadautorregresiva. Implementamos
estodentrodelaatencióndelproductoescaladoenmascarando(estableciendoen∞)todoslosvaloresenla
entradadelsoftmaxquecorrespondenaconexionesilegales. VerFigura2.
3.3 Redesderetroalimentaciónporposición
Ademásdelassubcapasdeatención,cadaunadelascapasdenuestrocodificadorydecodificadorcontieneunaredde
alimentacióndirectacompletamenteconectada,queseaplicaacadaposicióndeformaseparadaeidéntica. Consisteen
dostransformacioneslinealesconunaactivaciónReLUenelmedio.
FFN(x)=max(0,xW +b )W +b (2)
1 1 2 2
Sibienlastransformacioneslinealessonlasmismasendiferentesposiciones,utilizandiferentesparámetrosdeuna
capaaotra. Otraformadedescribirestoescomodosconvolucionescontamañodenúcleo1. Ladimensionalidadde
entradaysalidaesd =512,ylacapainternatienedimensionalidadd =2048.
model ff
3.4 IncrustacionesySoftmax
Demanerasimilaraotrosmodelosdetransduccióndesecuencias,utilizamosincrustacionesaprendidasparaconvertir
lostokensdeentradaylostokensdesalidaenvectoresdedimensióndmodel. Tambiénutilizamoslatransformación
linealaprendidahabitualylafunciónsoftmaxparaconvertirlasalidadeldecodificadorenprobabilidadespredichas
delsiguientetoken. Ennuestromodelo,compartimoslamismamatrizdepesoentrelasdoscapasdeincrustacióny
l√atransformaciónlinealpreviaasoftmax,similara[30]. Enlascapasdeincrustación,multiplicamosesospesospor
d .
model
AbandonoresidualAplicamosabandono[33]alasalidadecadasubcapa,antesdeagregarloalaentradadelasubcapa
ynormalizarlo. Además,aplicamosabandonoalassumasdelasincrustacionesylascodificacionesposicionalesenlas
pilasdecodificadorydecodificador. Paraelmodelobase,utilizamosunatasadeP =0,1.
drop
3.5 Codificaciónposicional
Dadoquenuestromodelonocontienerecurrencianiconvolución,paraqueelmodeloutiliceelordendelasecuencia,
debemosinyectaralgunainformaciónsobrelaposiciónrelativaoabsolutadelostokensenlasecuencia. Conestefin,
agregamos"codificacionesposicionales"alasincrustacionesdeentradaenlaparteinferiordelaspilasdecodificadores
5

LayerType ComplexityperLayer SequentialOperations MaximumPathLength
Self-Attention O(n2·d) O(1) O(1)
Recurrent O(n·d2) O(n) O(n)
Convolutional O(k·n·d2) O(1) O(log(n))
Self-Attention(restricted) O(r·n·d) O(1) O(n/r)
Table 1: Longitudes de ruta máximas, complejidad por capa y número mínimo de operaciones secuenciales para
diferentestiposdecapas. neslalongituddelasecuencia,desladimensiónderepresentación,k eseltamañodel
núcleodelasconvolucionesyreltamañodelavecindadenautoatenciónrestringida.
ydecodificadores. Lascodificacionesposicionalestienenlamismadimensiónd quelasincrustaciones,demodo
model
quelasdossepuedensumar. Haymuchasopcionesdecodificacionesposicionales,aprendidasyfijas[9].
Enestetrabajoutilizamosfuncionessenoycosenodediferentesfrecuencias:
PE pos,2i)=sin(pos/100002i/dmodel)
(
PE pos,2i+1)=cos(pos/100002i/dmodel)
(
dondeposeslaposiciónyiesladimensión. Esdecir,cadadimensióndelacodificaciónposicionalcorrespondeauna
sinusoide. Laslongitudesdeondaformanunaprogresióngeométricadesde2πhasta100002π. Elegimosestafunción
porqueplanteamoslahipótesisdequepermitiríaqueelmodeloaprendierafácilmenteaatenderporposicionesrelativas,
yaqueparacualquierdesplazamientofijok,P sepuederepresentarcomounafunciónlinealdePE .
Epos+k pos
Tambiénexperimentamosconelusodeincrustacionesposicionalesaprendidas[9]ydescubrimosquelasdosversiones
produjeronresultadoscasiidénticos(consultelafila(E)delaTabla3). Elegimoslaversiónsinusoidalporquepuede
permitirqueelmodeloextrapolalongitudesdesecuenciamáslargasquelasencontradasduranteelentrenamiento.
4 Porquélaautoatención
Enestaseccióncomparamosvariosaspectosdelascapasdeautoatenciónconlascapasrecurrentesyconvolucionales
comúnmenteutilizadasparamapearunasecuenciaderepresentacionesdesímbolosdelongitudvariable(x1,...,xn)a
otrasecuenciadeiguallongitud(z1,....,zn),conx ,z ϵRd,comounacapaocultaenuncodificadorodecodificador
i i
detransduccióndesecuenciatípico. Paramotivarnuestrousodelaautoatenciónconsideramostresdesiderata.
Unoeslacomplejidadcomputacionaltotalporcapa. Otraeslacantidaddecálculoquesepuedeparalelizar,medidapor
elnúmeromínimodeoperacionessecuencialesrequeridas.
Elterceroeslalongituddelcaminoentredependenciasdelargoalcanceenlared. Aprenderdependenciasdelargo
alcanceesundesafíoclaveenmuchastareasdetransduccióndesecuencias. Unfactorclavequeafectalacapacidadde
aprendertalesdependenciaseslalongituddeloscaminosquedebenrecorrerlasseñaleshaciaadelanteyhaciaatrásen
lared. Cuantomáscortosseanestoscaminosentrecualquiercombinacióndeposicionesenlassecuenciasdeentrada
ysalida,másfácilseráaprenderdependenciasdelargoalcance[12]. Porlotanto,tambiéncomparamoslalongitud
máximadelarutaentredosposicionesdeentradaysalidacualesquieraenredescompuestaspordiferentestiposde
capas.
Como se indica en la Tabla 1, una capa de autoatención conecta todas las posiciones con un número constante de
operacionesejecutadassecuencialmente,mientrasqueunacaparecurrenterequiereO(n)operacionessecuenciales. En
términosdecomplejidadcomputacional,lascapasdeautoatenciónsonmásrápidasquelascapasrecurrentescuando
la longitud de la secuencia n es menor que la dimensionalidad de representación d, que es el caso más frecuente
conlasrepresentacionesdeoracionesutilizadasporlosmodelosmásmodernosentraduccionesautomáticas,como
representacionesdefragmentosdepalabras[38]yparesdebytes[31]. Paramejorarelrendimientocomputacional
paratareasqueinvolucransecuenciasmuylargas,laautoatenciónpodríarestringirseaconsiderarsolounavecindadde
tamañorenlasecuenciadeentradacentradaalrededordelaposicióndesalidarespectiva. Estoaumentaríalalongitud
máximadelcaminoaO(n/r). Planeamosinvestigaresteenfoquemásafondoentrabajosfuturos.
Unaúnicacapaconvolucionalconunanchodenúcleok <nnoconectatodoslosparesdeposicionesdeentradaysalida.
HacerlorequiereunapiladecapasconvolucionalesO(n/k)enelcasodenúcleoscontiguos,uO(logk(n))enelcaso
deconvolucionesdilatadas[18],aumentandolalongituddeloscaminosmáslargosentredosposicionescualesquiera.
enlared. Lascapasconvolucionalessongeneralmentemáscarasquelascapasrecurrentes,porunfactordek. Las
6

convolucionesseparables[6],sinembargo,disminuyenconsiderablementelacomplejidad,aO(k·n·d+n·d2). Sin
embargo,inclusoconk =n,lacomplejidaddeunaconvoluciónseparableesigualalacombinacióndeunacapade
autoatenciónyunacapaderetroalimentaciónpuntual,elenfoquequeadoptamosennuestromodelo.
Comobeneficioadicional,laautoatenciónpodríagenerarmodelosmásinterpretables.Inspeccionamoslasdistribuciones
deatencióndenuestrosmodelosypresentamosyanalizamosejemplosenelapéndice. Lascabezasdeatenciónindivid-
ualesnosóloaprendenclaramentearealizardiferentestareas,sinoquemuchasparecenexhibiruncomportamiento
relacionadoconlaestructurasintácticaysemánticadelasoraciones.
5 Entrenamiento
Estaseccióndescribeelrégimendeentrenamientoparanuestrosmodelos.
5.1 Datosdeentrenamientoyprocesamientoporlotes
Nosentrenamosconelconjuntodedatosestándaringlés-alemánWMT2014queconstadeaproximadamente4,5
millonesdeparesdeoraciones. Lasoracionessecodificaronutilizandocodificacióndeparesdebytes[3],quetieneun
vocabulariofuente-destinocompartidodeaproximadamente37000tokens. Parainglés-francés,utilizamoselconjunto
dedatosinglés-francésWMT2014,significativamentemásgrande,queconstade36millonesdeoracionesytokens
divididosenunvocabulariode32000palabras[38]. Losparesdeoracionesseagruparonporlongituddesecuencia
aproximada. Cadalotedeentrenamientoconteníaunconjuntodeparesdeoracionesqueconteníanaproximadamente
25000tokensdeorigeny25000tokensdedestino.
5.2 Hardwareyprogramación
Entrenamosnuestrosmodelosenunamáquinacon8GPUNVIDIAP100. Paranuestrosmodelosbasequeutilizanlos
hiperparámetrosdescritosalolargodelartículo,cadapasodeentrenamientotomóaproximadamente0,4segundos.
Entrenamoslosmodelosbaseparauntotalde100.000pasoso12horas. Paranuestrosmodelosgrandes,descritosenla
últimalíneadelatabla3),eltiempodepasofuede1,0segundos. Losgrandesmodelosfueronentrenadospara300.000
pasos(3,5días).
5.3 Optimizador
UsamoseloptimizadorAdam[20]conβ =0.9,β =0.98yϵ=10−9. Variamoslatasadeaprendizajealolargode
1 2
laformación,segúnlafórmula:
lrate=d−0.5 ·min(step_num−0.5,step_num·warmup_steps−1.5) (3)
model
Esto corresponde a aumentar la tasa de aprendizaje linealmente para los primeros pasos de entrenamiento de
warmup tepsydisminuirlaposteriormenteproporcionalmentealaraízcuadradainversadelnúmerodepaso. Usamos
s
warmup teps=4000.
s
5.4 Regularización
Empleamostrestiposderegularizacióndurantelaformación:
SuavizadodeetiquetasDuranteelentrenamiento,empleamosunsuavizadodeetiquetasdevalorϵ =0,1[36]. Esto
ls
duelelaperplejidad,yaqueelmodeloaprendeasermásinseguro,peromejoralaprecisiónylapuntuaciónBLEU.
AbandonoresidualAplicamosabandono[33]alasalidadecadasubcapa,antesdeagregarloalaentradadelasubcapa
ynormalizarlo. Además,aplicamosabandonoalassumasdelasincrustacionesylascodificacionesposicionalesenlas
pilasdecodificadorydecodificador. Paraelmodelobase,utilizamosunatasadeP =0.1.
drop
SuavizadodeetiquetasDuranteelentrenamiento,empleamosunsuavizadodeetiquetasdevalorϵ =0.1[36]. Esto
ls
duelelaperplejidad,yaqueelmodeloaprendeasermásinseguro,peromejoralaprecisiónylapuntuaciónBLEU.
7

| Model                       | BLEU  |       | TrainingCost(FLOPs) |          |
| --------------------------- | ----- | ----- | ------------------- | -------- |
|                             | EN-DE | EN-FR | EN-DE               | EN-FR    |
| ByteNet[18]                 | 23.75 |       |                     |          |
| Deep-Att+PosUnk[39]         |       | 39.92 |                     | 1.4·1020 |
| GNMT+RL[38]                 | 24.6  | 39.92 | 9.6·1018            | 1.5·1020 |
|                             |       |       | 2.0·1019            | 1.2·1020 |
| ConvS2S[9]                  | 25.16 | 40.46 |                     |          |
| MoE[32]                     | 26.03 | 40.56 | 8.0·1019            | 1.2·1020 |
| Deep-Att+PosUnkEnsemble[39] |       | 40.4  |                     | 1.2·1021 |
|                             |       |       | 7.7·1019            | 1.2·1021 |
| GNMT+RLEnsemble[38]         | 26.30 | 41.29 |                     |          |
| ConvS2SEnsemble[9]          | 26.26 | 41.29 | 1.8·1020            | 1.2·1021 |
| Transformer(basemodel)      | 27.3  | 38.1  | 3.3·1018            |          |
2.3·1019
| Transformer(big) | 28.4 | 41.8 |     |     |
| ---------------- | ---- | ---- | --- | --- |
Table2: ElTransformerlogramejorespuntajesBLEUquelosmodelosanterioresdeúltimageneraciónenlaspruebas
Newstest2014deinglésaalemáneinglésafrancésaunafraccióndelcostodecapacitación.
6 Resultados
6.1 Traducciónautomática
EnlatareadetraduccióndelinglésalalemándelWMT2014,elmodelodetransformadorgrande(Transformador
(grande)enlaTabla2)superaalosmejoresmodelosreportadosanteriormente(incluidoslosconjuntos)enmásde2.0
BLEU,estableciendounnuevoestadodelasituación. PuntuaciónartísticaBLEUde28,4. Laconfiguracióndeeste
modeloseenumeraenlalíneainferiordelaTabla3. Elentrenamientotomó3.5díasen8GPUP100. Inclusonuestro
modelobasesuperatodoslosmodelosyconjuntospublicadosanteriormente,aunafraccióndelcostodecapacitación
decualquieradelosmodeloscompetitivos.
EnlatareadetraduccióndelinglésalfrancésdelWMT2014,nuestromodelograndelograunapuntuaciónBLEUde
41.0,superandoatodoslosmodelosindividualespublicadosanteriormente,amenosde1/4delcostodecapacitacióndel
estadodelarteanterior. ElmodeloTransformer(grande)entrenadoparainglésafrancésutilizóunatasadedeserción
Pdrop=0.1,enlugarde0.3.
Paralosmodelosbase, utilizamosunmodeloúnicoobtenidopromediandolosúltimos5puntosdecontrol, quese
escribieronenintervalosde10minutos. Paralosmodelosgrandes,promediamoslosúltimos20puntosdecontrol.
Utilizamos búsqueda de haz con un tamaño de haz de 4 y una penalización de longitud ∞ = 0,6 [38]. Estos
hiperparámetrosseeligierondespuésdeexperimentarenelconjuntodedesarrollo. Establecemoslalongitudmáxima
desalidadurantelainferenciaenlalongituddeentrada+50,peroterminamostempranocuandoesposible[38].
LaTabla2resumenuestrosresultadosycomparanuestracalidaddetraducciónycostosdecapacitaciónconotras
arquitecturasmodelodelaliteratura. Estimamoslacantidaddeoperacionesdepuntoflotanteutilizadasparaentrenar
unmodelomultiplicandoeltiempodeentrenamiento,lacantidaddeGPUutilizadasyunaestimacióndelacapacidad
sostenidadepuntoflotantedeprecisiónsimpledecadaGPU55.
6.2 Variacionesdelmodelo
ParaevaluarlaimportanciadelosdiferentescomponentesdelTransformer,variamosnuestromodelobasedediferentes
maneras, midiendoelcambioenelrendimientoenlatraduccióndelinglésalalemánenelconjuntodedesarrollo,
newstest2013. Utilizamoslabúsquedadehazcomosedescribeenlasecciónanterior,peronopromediamoselpuntode
control. PresentamosestosresultadosenlaTabla3.
EnlasfilasdelaTabla3(A),variamoselnúmerodecabezasdeatenciónylasdimensionesclaveydevalordeatención,
manteniendoconstantelacantidaddecálculo,comosedescribeenlaSección3.2.2. Sibienlaatenciónconunsolo
cabezales0.9BLEUpeorquelamejorconfiguración,lacalidadtambiéndisminuyecondemasiadoscabezales.
EnlasfilasdelaTabla3(B),observamosquereducireltamañodelaclavedeatenciónd perjudicalacalidaddel
k
modelo. Estosugierequedeterminarlacompatibilidadnoesfácilyqueunafuncióndecompatibilidadmássofisticada
queelproductoescalarpuedeserbeneficiosa. Además,observamosenlasfilas(C)y(D)que,comoseesperaba,los
modelosmásgrandessonmejoresyqueelabandonoesmuyútilparaevitarelsobreajuste. Enlafila(E)reemplazamos
5Utilizamosvaloresde2,8,3,7,6,0y9,5TFLOPSparaK80,K40,M40yP100,respectivamente.
8

nuestracodificaciónposicionalsinusoidalconincrustacionesposicionalesaprendidas[9]yobservamosresultadoscasi
idénticosalmodelobase.
|      | N d d      | h d d     | P ϵ     | trainsteps | PPL BLEU    | params |
| ---- | ---------- | --------- | ------- | ---------- | ----------- | ------ |
|      | model ff   | k v       | drop ls |            |             |        |
|      |            |           |         |            | (dev) (dev) | ×106   |
| Base | 6 512 2048 | 8 64 64   | 0.1 0.1 | 100K       | 4.92 25.8   | 65     |
|      |            | 1 512 512 |         |            | 5.29 24.9   |        |
|      |            | 4 128 128 |         |            | 5.00 25.5   |        |
(A)
|     |     | 16 32 32 |     |     | 4.91 25.8 |     |
| --- | --- | -------- | --- | --- | --------- | --- |
|     |     | 32 16 16 |     |     | 5.01 25.4 |     |
|     |     | 16       |     |     | 5.16 25.1 | 58  |
(B)
|     |      | 32      |     |     | 5.01 25.4 | 60  |
| --- | ---- | ------- | --- | --- | --------- | --- |
|     | 2    |         |     |     | 6.11 23.7 |     |
|     | 4    |         |     |     | 5.19 25.3 |     |
|     | 8    |         |     |     | 4.88 25.5 |     |
| (C) | 256  | 32 32   |     |     | 5.75 24.5 |     |
|     | 1024 | 128 128 |     |     | 4.66 26.0 |     |
|     | 1024 |         |     |     | 5.12 25.4 |     |
|     | 4096 |         |     |     | 4.75 26.2 |     |
|     |      |         | 0.0 |     | 5.77 24.6 |     |
|     |      |         | 0.2 |     | 4.95 25.5 |     |
(D)
|     |                                       |     | 0.0 |      | 4.67 25.3 |     |
| --- | ------------------------------------- | --- | --- | ---- | --------- | --- |
|     |                                       |     | 0.2 |      | 5.47 25.7 |     |
| (E) | positionalembeddinginsteadofsinusoids |     |     |      | 4.92 25.7 |     |
| big | 6 1024 4096                           | 16  | 0.3 | 300K | 4.33 26.4 | 213 |
Table 3: Variaciones sobre la arquitectura Transformer. Los valores no listados son idénticos a los del modelo
base. Todaslasmétricasestánenelconjuntodedesarrollodetraduccióndelinglésalalemán, newstest2013. Las
perplejidadesenumeradassonporpiezadepalabra,segúnnuestracodificacióndeparesdebytes,ynodebencompararse
conperplejidadesporpalabra.
6.3 Análisisdecircunscripcionesinglesas
ParaevaluarsielTransformerpuedegeneralizarseaotrastareas,realizamosexperimentosenelanálisisdedistritos
electoraleseninglés.Estatareapresentadesafíosespecíficos:elresultadoestásujetoafuertesrestriccionesestructurales
yessignificativamentemáslargoqueelinsumo. Además,losmodelosRNNsecuenciaasecuencianohanpodido
lograrresultadosdeúltimageneraciónenregímenesdedatospequeños[37].
Entrenamos un transformador de 4 capas con d model = 1024 en la parte del Wall Street Journal (WSJ) del Penn
Treebank[25],alrededorde40.000oracionesdeentrenamiento. Tambiénloentrenamosenunentornosemisupervisado,
utilizandocorpusmásgrandesdealtaconfianzayBerkleyParserconaproximadamente17millonesdeoraciones[37].
Usamosunvocabulariode16KtokensparalaconfiguracióndeWSJúnicamenteyunvocabulariode32Ktokenspara
laconfiguraciónsemisupervisada.
Realizamossolounapequeñacantidaddeexperimentosparaseleccionarladeserción,tantolaatencióncomolaresidual
(sección5.4),lastasasdeaprendizajeyeltamañodelhazenelconjuntodedesarrollodelaSección22;todoslosdemás
parámetrossemantuvieronsincambiosconrespectoalmodelobasedetraduccióndelinglésalalemán. Durantela
inferencia,aumentamoslalongitudmáximadesalidaalalongituddeentrada+300. Utilizamosuntamañodehazde
21y∞=0.3tantoparaWSJúnicamentecomoparalaconfiguraciónsemisupervisada.
NuestrosresultadosenlaTabla4muestranqueapesardelafaltadeajusteespecíficodelatarea, nuestromodelo
funcionasorprendentementebien,arrojandomejoresresultadosquetodoslosmodelosinformadosanteriormente,con
laexcepcióndelagramáticaderedesneuronalesrecurrentes[8].
AdiferenciadelosmodelossecuenciaasecuenciaRNN[37],TransformersuperaaBerkeleyParser[29]inclusocuando
seentrenasoloenelconjuntodeentrenamientoWSJde40Koraciones.
9

| Parser                 |            | Training               | WSJ23F1 |
| ---------------------- | ---------- | ---------------------- | ------- |
| Vinyals&Kaiserelal.    | (2014)[37] | WSJonly,discriminative | 88.3    |
| Petrovetal.            | (2006)[29] | WSJonly,discriminative | 90.4    |
| Zhuetal.               | (2013)[40] | WSJonly,discriminative | 90.4    |
| Dyeretal.              | (2016)[8]  | WSJonly,discriminative | 91.7    |
| Transformer(4layers)   |            | WSJonly,discriminative | 91.3    |
| Zhuetal.               | (2013)[40] | semi-supervised        | 91.3    |
| Huang&Harper(2009)[14] |            | semi-supervised        | 91.3    |
| McCloskyetal.          | (2006)[26] | semi-supervised        | 92.1    |
| Vinyals&Kaiserelal.    | (2014)[37] | semi-supervised        | 92.1    |
| Transformer(4layers)   |            | semi-supervised        | 92.7    |
| Luongetal.             | (2015)[23] | multi-task             | 93.0    |
| Dyeretal.              | (2016)[8]  | generative             | 93.3    |
Table4: TheTransformersegeneralizabienalanálisisdedistritoselectoraleseninglés(losresultadosseencuentranen
laSección23delWSJ).
7 Conclusión
Enestetrabajo,presentamosTransformer,elprimermodelodetransduccióndesecuenciabasadocompletamenteenla
atención,reemplazandolascapasrecurrentesmáscomúnmenteutilizadasenarquitecturasdecodificador-decodificador
conautoatencióndemúltiplescabezas.
Paratareasdetraducción,Transformersepuedeentrenarsignificativamentemásrápidoquelasarquitecturasbasadas
encapasrecurrentesoconvolucionales. TantoenlastareasdetraduccióndeinglésaalemándeWMT2014comode
inglésafrancésdeWMT2014,logramosunnuevoestadodelarte. Enlaprimeratarea,nuestromejormodelosupera
inclusoatodoslosconjuntosreportadosanteriormente.
Estamos entusiasmados con el futuro de los modelos basados en la atención y planeamos aplicarlos a otras tareas.
Planeamos extender Transformer a problemas que involucran modalidades de entrada y salida distintas al texto e
investigarmecanismoslocalesdeatenciónrestringidaparamanejareficientementegrandesentradasysalidas,como
imágenes,audioyvideo. Hacerquelageneraciónseamenossecuencialesotrodenuestrosobjetivosdeinvestigación.
El código que utilizamos para entrenar y evaluar nuestros modelos está disponible en https://github.com/
tensorflow/tensor2tensor. AgradecimientosAgradecemosaNalKalchbrenneryStephanGouwsporsusfruc-
tíferoscomentarios,correccioneseinspiración.
References
[1] JimmyLeiBa,JamieRyanKiros,andGeoffreyE.Hinton. Layernormalization,2016.
[2] DzmitryBahdanau,KyunghyunCho,andYoshuaBengio. Neuralmachinetranslationbyjointlylearningtoalign
andtranslate,2016.
[3] DennyBritz,AnnaGoldie,Minh-ThangLuong,andQuocLe. Massiveexplorationofneuralmachinetranslation
architectures. InMarthaPalmer,RebeccaHwa,andSebastianRiedel,editors,Proceedingsofthe2017Conference
onEmpiricalMethodsinNaturalLanguageProcessing,pages1442–1451,Copenhagen,Denmark,September
2017.AssociationforComputationalLinguistics.
[4] JianpengCheng,LiDong,andMirellaLapata.Longshort-termmemory-networksformachinereading.InJianSu,
KevinDuh,andXavierCarreras,editors,Proceedingsofthe2016ConferenceonEmpiricalMethodsinNatural
LanguageProcessing,pages551–561,Austin,Texas,November2016.AssociationforComputationalLinguistics.
[5] KyunghyunCho,BartvanMerriënboer,CaglarGulcehre,DzmitryBahdanau,FethiBougares,HolgerSchwenk,
and Yoshua Bengio. Learning phrase representations using RNN encoder–decoder for statistical machine
translation.InAlessandroMoschitti,BoPang,andWalterDaelemans,editors,Proceedingsofthe2014Conference
onEmpiricalMethodsinNaturalLanguageProcessing(EMNLP),pages1724–1734,Doha,Qatar,October2014.
AssociationforComputationalLinguistics.
[6] FrançoisChollet. Xception: Deeplearningwithdepthwiseseparableconvolutions. CoRR,abs/1610.02357,2016.
[7] JunyoungChung,CaglarGulcehre,KyungHyunCho,andYoshuaBengio. Empiricalevaluationofgatedrecurrent
neuralnetworksonsequencemodeling,2014.
10

[8] ChrisDyer,AdhigunaKuncoro,MiguelBallesteros,andNoahA.Smith. Recurrentneuralnetworkgrammars.
InKevinKnight,AniNenkova,andOwenRambow,editors,Proceedingsofthe2016ConferenceoftheNorth
American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages
199–209,SanDiego,California,June2016.AssociationforComputationalLinguistics.
[9] JonasGehring,MichaelAuli,DavidGrangier,DenisYarats,andYannN.Dauphin. Convolutionalsequenceto
sequencelearning,2017.
[10] AlexGraves. Generatingsequenceswithrecurrentneuralnetworks. CoRR,abs/1308.0850,2013.
[11] KaimingHe,XiangyuZhang,ShaoqingRen,andJianSun. Deepresiduallearningforimagerecognition. In2016
IEEEConferenceonComputerVisionandPatternRecognition(CVPR),pages770–778,2016.
[12] S. Hochreiter, Y. Bengio, P. Frasconi, and J. Schmidhuber. Gradient flow in recurrent nets: the difficulty of
learninglong-termdependencies. InS.C.KremerandJ.F.Kolen,editors,AFieldGuidetoDynamicalRecurrent
NeuralNetworks.IEEEPress,2001.
[13] SeppHochreiterandJürgenSchmidhuber. LongShort-TermMemory. NeuralComputation,9(8):1735–1780,11
1997.
[14] ZhongqiangHuangandMaryHarper. Self-trainingPCFGgrammarswithlatentannotationsacrosslanguages. In
PhilippKoehnandRadaMihalcea,editors,Proceedingsofthe2009ConferenceonEmpiricalMethodsinNatural
LanguageProcessing,pages832–841,Singapore,August2009.AssociationforComputationalLinguistics.
[15] RafalJózefowicz,OriolVinyals,MikeSchuster,NoamShazeer,andYonghuiWu.Exploringthelimitsoflanguage
modeling. CoRR,abs/1602.02410,2016.
[16] ŁukaszKaiserandSamyBengio. Canactivememoryreplaceattention? InD.Lee,M.Sugiyama,U.Luxburg,
I. Guyon, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 29. Curran
Associates,Inc.,2016.
[17] LukaszKaiserandIlyaSutskever. Neuralgpuslearnalgorithms. arXiv: Learning,2015.
[18] Nal Kalchbrenner, Lasse Espeholt, Karen Simonyan, Aäron van den Oord, Alexander Graves, and Koray
Kavukcuoglu. Neuralmachinetranslationinlineartime. 2016.
[19] YoonKim,CarlDenton,LuongHoang,andAlexanderM.Rush. Structuredattentionnetworks. InInternational
ConferenceonLearningRepresentations,2017.
[20] DiederikP.KingmaandJimmyBa. Adam: Amethodforstochasticoptimization,2017.
[21] OleksiiKuchaievandBorisGinsburg. Factorizationtricksforlstmnetworks,2018.
[22] ZhouhanLin,MinweiFeng,CiceroNogueiradosSantos,MoYu,BingXiang,BowenZhou,andYoshuaBengio.
ASTRUCTUREDSELF-ATTENTIVESENTENCEEMBEDDING. InInternationalConferenceonLearning
Representations,2017.
[23] Minh-Thang Luong, Quoc V Le, Ilya Sutskever, Oriol Vinyals, and Lukasz Kaiser. Multi-task sequence to
sequencelearning. arXivpreprintarXiv:1511.06114,2015.
[24] ThangLuong,HieuPham,andChristopherD.Manning. Effectiveapproachestoattention-basedneuralmachine
translation. InLluísMàrquez,ChrisCallison-Burch,andJianSu,editors,Proceedingsofthe2015Conference
onEmpiricalMethodsinNaturalLanguageProcessing,pages1412–1421,Lisbon,Portugal,September2015.
AssociationforComputationalLinguistics.
[25] Mitchell P. Marcus, Beatrice Santorini, and Mary Ann Marcinkiewicz. Building a large annotated corpus of
English: ThePennTreebank. ComputationalLinguistics,19(2):313–330,1993.
[26] DavidMcClosky,EugeneCharniak,andMarkJohnson. Effectiveself-trainingforparsing. InRobertC.Moore,
JeffBilmes,JenniferChu-Carroll,andMarkSanderson,editors,ProceedingsoftheHumanLanguageTechnology
ConferenceoftheNAACL,MainConference,pages152–159,NewYorkCity,USA,June2006.Associationfor
ComputationalLinguistics.
[27] AnkurParikh,OscarTäckström,DipanjanDas,andJakobUszkoreit. Adecomposableattentionmodelfornatural
languageinference. InJianSu,KevinDuh,andXavierCarreras,editors,Proceedingsofthe2016Conference
on Empirical Methods in Natural Language Processing, pages 2249–2255, Austin, Texas, November 2016.
AssociationforComputationalLinguistics.
[28] RomainPaulus,CaimingXiong,andRichardSocher. Adeepreinforcedmodelforabstractivesummarization,
2017.
11

[29] SlavPetrov,LeonBarrett,RomainThibaux,andDanKlein. Learningaccurate,compact,andinterpretabletree
annotation. InNicolettaCalzolari,ClaireCardie,andPierreIsabelle,editors,Proceedingsofthe21stInterna-
tionalConferenceonComputationalLinguisticsand44thAnnualMeetingoftheAssociationforComputational
Linguistics,pages433–440,Sydney,Australia,July2006.AssociationforComputationalLinguistics.
[30] OfirPressandLiorWolf. Usingtheoutputembeddingtoimprovelanguagemodels. InMirellaLapata, Phil
Blunsom,andAlexanderKoller,editors,Proceedingsofthe15thConferenceoftheEuropeanChapterofthe
AssociationforComputationalLinguistics: Volume2,ShortPapers,pages157–163,Valencia,Spain,April2017.
AssociationforComputationalLinguistics.
[31] RicoSennrich,BarryHaddow,andAlexandraBirch. Neuralmachinetranslationofrarewordswithsubword
units. InKatrinErkandNoahA.Smith, editors, Proceedingsofthe54thAnnualMeetingoftheAssociation
for Computational Linguistics (Volume 1: Long Papers), pages 1715–1725, Berlin, Germany, August 2016.
AssociationforComputationalLinguistics.
[32] NoamShazeer,*AzaliaMirhoseini,*KrzysztofMaziarz,AndyDavis,QuocLe,GeoffreyHinton,andJeffDean.
Outrageouslylargeneuralnetworks: Thesparsely-gatedmixture-of-expertslayer. InInternationalConferenceon
LearningRepresentations,2017.
[33] NitishSrivastava,GeoffreyHinton,AlexKrizhevsky,IlyaSutskever,andRuslanSalakhutdinov.Dropout:Asimple
waytopreventneuralnetworksfromoverfitting. JournalofMachineLearningResearch,15(56):1929–1958,
2014.
[34] SainbayarSukhbaatar,arthurszlam,JasonWeston,andRobFergus. End-to-endmemorynetworks. InC.Cortes,
N.Lawrence,D.Lee,M.Sugiyama,andR.Garnett,editors,AdvancesinNeuralInformationProcessingSystems,
volume28.CurranAssociates,Inc.,2015.
[35] IlyaSutskever,OriolVinyals,andQuocVLe. Sequencetosequencelearningwithneuralnetworks. InZ.Ghahra-
mani, M. Welling, C. Cortes, N. Lawrence, and K.Q. Weinberger, editors, Advances in Neural Information
ProcessingSystems,volume27.CurranAssociates,Inc.,2014.
[36] ChristianSzegedy, VincentVanhoucke, SergeyIoffe, JonathonShlens, andZbigniewWojna. Rethinkingthe
inceptionarchitectureforcomputervision,2015.
[37] Oriol Vinyals, Ł ukasz Kaiser, Terry Koo, Slav Petrov, Ilya Sutskever, and Geoffrey Hinton. Grammar as a
foreignlanguage. InC.Cortes,N.Lawrence,D.Lee,M.Sugiyama,andR.Garnett,editors,AdvancesinNeural
InformationProcessingSystems,volume28.CurranAssociates,Inc.,2015.
[38] Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi, Wolfgang Macherey, Maxim
Krikun,YuanCao,QinGao,KlausMacherey,JeffKlingner,ApurvaShah,MelvinJohnson,XiaobingLiu,Łukasz
Kaiser,StephanGouws,YoshikiyoKato,TakuKudo,HidetoKazawa,KeithStevens,GeorgeKurian,Nishant
Patil,WeiWang,CliffYoung,JasonSmith,JasonRiesa,AlexRudnick,OriolVinyals,GregCorrado,Macduff
Hughes,andJeffreyDean. Google’sneuralmachinetranslationsystem: Bridgingthegapbetweenhumanand
machinetranslation,2016.
[39] JieZhou,YingCao,XuguangWang,PengLi,andWeiXu. Deeprecurrentmodelswithfast-forwardconnections
forneuralmachinetranslation. TransactionsoftheAssociationforComputationalLinguistics,4:371–383,2016.
[40] MuhuaZhu,YueZhang,WenliangChen,MinZhang,andJingboZhu. Fastandaccurateshift-reduceconstituent
parsing. InHinrichSchuetze,PascaleFung,andMassimoPoesio,editors,Proceedingsofthe51stAnnualMeeting
of the Association for Computational Linguistics (Volume 1: Long Papers), pages 434–443, Sofia, Bulgaria,
August2013.AssociationforComputationalLinguistics.
12

Visualizacionesdeatención
Figure 3: Un ejemplo del mecanismo de atención que sigue dependencias a larga distancia en la autoatención del
codificadorenlacapa5de6. Muchasdelascabezasdeatenciónatiendenaunadependenciadistantedelverbo"making
(hacer)",completandolafrase"making...moredifficult". Lasatencionesaquísemuestransóloparalapalabra"making
(hacer)". Diferentescoloresrepresentandiferentescabezas. Mejorvistoencolor.
13

Figure4: Doscabezasdeatención,tambiénenlacapa5de6,aparentementeinvolucradasenlaresolucióndelaanáfora.
Arriba: atenciones completas para los encabezados 5. Abajo: atenciones aisladas solo de la palabra "its" para los
encabezadosdeatención5y6. Tengaencuentaquelasatencionessonmuynítidasparaestapalabra.
14

Figure5: Muchasdelascabezasdeatenciónexhibenuncomportamientoqueparecerelacionadoconlaestructuradela
oración. Arribadamosdosejemplosdeestetipo,dedoscabezalesdiferentesdelaautoatencióndelcodificadorenla
capa5de6. Loscabezalesclaramenteaprendieronarealizardiferentestareas.
15