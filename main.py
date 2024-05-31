import argparse
import json
import gzip
import os
import numpy as np

import pandas as pd
import numpy as np
import spacy
import ast
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from lemmagen3 import Lemmatizer
import difflib
import os


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
import random
from sklearn.preprocessing import OneHotEncoder
#dla, na gpu
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import time
import json
import re
nlp = spacy.load("sl_core_news_trf")




slo_stopwords = ['petindvajsetemu', 'nekakih', 'skorajda', 'kdorkoli', 'osemindevetdesetim', 'osemdesetim', 'morete', 'šestindvajsetih', 'nekoč', 'stotero', 'sami', 'le-taka', 'eden', 'izpod', 'sta', 'trideseta', 'našima', 'tretjih', 'enkratnem', 'kakršnakoli', 'štiriindvajsetimi', 'tisto', 'cel', 'pa', 'trinajstem', 'drugačnega', 'sedemnajstima', 'vsaka', 'nekaki', 'nekateri', 'drugačnim', 'rad', 'težki', 'štiridesetimi', 'želiš', 'dvajset', 'vrh', 'svojega', 'bodita', 'enkratnih', 'tolikšnim', 'iste', 'kolikimi', 'smeš', 'šestimi', 'tisočih', 'petinosemdesetim', 'deseterimi', 'dvojna', 'deseterima', 'stotem', 'četrtim', 'prvimi', 'enaindvajseto', 'troja', 'trojo', 'kakršni', 'dvainšestdesetim', 'skozi', 'moramo', 'takile', 'morale', 'hočeva', 'skozenj', 'maralo', 'dolga', 'nečesa', 'vii', 'štiridesetega', 'marsikaterim', 'istim', 'bržkone', 'šestnajstih', 'nekaterem', 'takemule', 'stoterem', 'sedemindvajsetem', 'svojem', 'le-onimi', 'njegova', 'takega', 'njihovih', 'one', 'jih', 'isto', 'štiristotih', 'nekake', 'le-onim', 'lepo', 'njunem', 'od', 'petindvajset', 'devetega', 'gospod', 'nobenih', 'redko', 'dvaindevetdesetega', 'sedemindvajsetemu', 'nekakšnih', 'mednju', 'dvanajstima', 'vajina', 'nas', 'le-takšno', 'oktober', 'f', 'dvestota', 'katerekoli', 'rada', 'svoja', 'kaka', 'prvi', 'odkar', 'in', 'kakršnem', 'si', 'njegovi', 'katerimkoli', 'pozdravljen', 'kratki', 'tretjemu', 'četudi', 'njiju', 'štiristotimi', 'namesto', 'tretjimi', 'tolikimi', 'morem', 'onem', 'enakemu', 'dvakratnimi', 'kolikšni', 'pome', 'dobro', 'trinajstih', 'vaši', 'nekaterima', 'nad', 'izza', 'dvanajstemu', 'nekakšnega', 'kakršnemkoli', 'stran', 'dvakratnima', 'čimer', 'kogar', 'etc.', 'komu', 'čemer', 'nekoliko', 'cela', 'zmogel', 'dvestotih', 'čeravno', 'njena', 'kakršnega', 'tisočo', 'enaintridesetih', 'bosti', 'sedmo', 'devetdeseti', 'kolikšnih', 'vsakršno', 'devetdesetimi', 'vsi', 'petinosemdesetega', 'deset', 'triintridesetih', 'kakor', 'vajinimi', 'petinosemdesetemu', 'kateregakoli', 'vaše', 'petintridesete', 'drugimi', 'le-takimi', 'petinsedemdeset', 'januar', 'marsikateri', 'vsakršnima', 'drugačni', 'vidva', 'enajst', 'kakem', 'kolikšnem', 'štiriindvajsetima', 'vaših', 'deveto', 'neki', 'enaindvajsetem', 'hotel', 'sedeminšestdeset', 'zmogle', 'le-takšnemu', 'polno', 'kakemu', 'najin', 'kakšno', 'raz', 'mesec', 'peterimi', 'nekoga', 'tridesetemu', 'tisoča', 'trojna', 'skoznjo', 'nikomer', 'štirinajstih', 'prednje', 'se', 'enem', 'petintridesetima', 'take', 'enkratne', 'osemnajstima', 'marsičim', 'njenem', 'tretjima', 'četrtem', 'njegovega', 'zmoreva', 'desetim', 'manj', 'zopet', 'štirinajstima', 'le-tak', 'nama', 'nikakršen', 'deseti', 'name', 'takle', 'hotiva', 'nato', 'morali', 'bila', 'dovolite', 'vso', 'le-takšnima', 'tridesete', 'triindvajseto', 'jesti', 'da', 'kadarkoli', 'sedemindvajseto', 'pozdravljeni', 'kakršna', 'obe', 'triintridesetemu', 'vsakih', 'deseterega', 'sedem', 'viia', 'njegovih', 'tisočega', 'spod', 'mojo', 'po', 'nekakimi', 'stvar', 'maraj', 'zanju', 'sebe', 'komer', 'nazaj', 'troj', 'osemnajstem', 'vašo', 'dvojima', 'le-takšnimi', 'obenj', 'le-one', 'dobesedno', 'marsikatere', 'razen', 'njenim', 'smemo', 'najinega', 'njegovem', 'enajste', 'čezenj', 'dober', 'lepe', 'katerihkoli', 'nedelja', 'temuintemu', 'tolikšnemu', 'vi', 'nobenem', 'katerokoli', 'naši', 'itak', 'nobenimi', 'njegovemu', 'preko', 'petinštirideseta', 'sedemdesetemu', 'devetdeset', 'triintridesete', 'vsemi', 'njegovimi', 'mojega', 'devetdesetem', 'trideseti', 'praviti', 'peteremu', 'kakšne', 'z', 'zame', 'petinštiridesetemu', 'osmi', 'trojega', 'oboj', 'njegovima', 'pripravljen', 'tvoja', 'triindvajsetima', 'more', 'predenje', 'lepa', 'sedmim', 'nekima', 'zavoljo', 'tukaj', 'trojno', 'niti', 'dvanajsti', 'šestnajst', 'trikraten', 'le-onemu', 'nočem', 'takele', 'takale', 'zmogl', 'našem', 'izmed', 'izpred', 'njegove', 'njo', 'kakršnih', 'šeststotimi', 'želela', 'prvih', 'marajva', 'triindvajsetemu', 'nikamor', 'nekih', 'šesto', 'ista', 'želeno', 'vajinim', 'desetima', 'podme', 'vseh', 'zadaj', 'moraš', 'temule', 'viii', 'le-tem', 'bomo', 'enakem', 'šestdesete', 'takšna', 've', 'približno', 'kakšen', 'takemu', 'sedma', 'marata', 'tvojih', 'le-tako', 'le-tista', 'njunemu', 'najinimi', 'vaša', 'sreda', 'katerimi', 'marsikaterimi', 'tj.', 'štiriindvajseta', 'njihove', 'dovolim', 'tiste', 'dvojnimi', 'bo', 'kajti', 'petintridesetim', 'triindvajsetih', 'katerimakoli', 'njunim', 'le-takšni', 'svojih', 'njuno', 'štiridesetima', 'nadme', 'njem', 'česa', 'bolj', 'kolikih', 'drugemu', 'petinpetdeset', 'takšnega', 'več', 'vse', 'le-takšnem', 'vata', 'meniti', 'vanje', 'toliko', 'triinpetdesetim', 'prave', 'obeh', 'le-tisti', 'dvojnem', 'nekakšna', 'osemnajstemu', 'devetemu', 'katerikoli', 'tolikih', 'tista', 'jima', 'koga', 'enega', 'kolikšno', 'vzdolž', 'nekaka', 'skoraj', 'h', 'devetnajstemu', 'šestintridesetim', 'petdeseto', 'istih', 'resda', 'šestih', 'tegale', 'le-tist', 'triinpetdesetimi', 'le-takem', 'enkratnim', 'dvestotega', 'sedmega', 'najinem', 'šestdeseto', 'nekak', 'devetdesetima', 'marajta', 'predme', 'tisočerem', 'vendarle', 'stoti', 'pod', 'peterima', 'enakima', 'enajstimi', 'desetemu', 'katerima', 'februar', 'prava', 'tej', 'dvakratno', 'vsako', 'dvanajstimi', 'petdesetem', 'le-onem', 'čimerkoli', 'tvojega', 'tisočim', 'le-oni', 'kakršnimakoli', 'petinosemdeseto', 'šestnajstima', 'petega', 'mu', 'šeststotim', 'tridesetimi', 'mano', 'nekako', 'dvaindevetdesetih', 'taka', 'trikratnimi', 'sedmem', 'smeti', 'triindvajseta', 'dvojo', 'tisočerima', 'osemdesetemu', 'smem', 'veliki', 'trinajsto', 'kakim', 'trikratne', 'onih', 'november', 'same', 'takimle', 'kamor', 'jem', 'marsikaterem', 'katerem', 'oseminštirideset', 'kake', 'predenj', 'svojo', 'kakršnim', 'dvestote', 'tristotimi', 'šestnajsto', 'lahki', 'tistimi', 'osem', 'drugem', 'takem', 'vsakdo', 'ene', 'kakršnemukoli', 'triinpetdesetih', 'nočejo', 'vsake', 'čezme', 'moj', 'bodisi', 'kakšnemu', 'hočem', 'dvanajst', 'trojnega', 'kakršnokoli', 'smejo', 'trinajstemu', 'trojih', 'triindvajsetim', 'sedmimi', 'sedemdesete', 'dovolil', 'čigavim', 'ob', 'triintridesetima', 'štirideseta', 'enakim', 'iznad', 'smeva', 'n', 'najine', 'nikjer', 'dvajseto', 'devetintridesetih', 'dvestotemu', 'nikakršnem', 'mogel', 'želeti', 'temile', 'tisoč', 'bojda', 'petdeseti', 'zmogla', 'sedemnajstim', 'ravno', 'dvojim', 'enajstim', 'povrh', 'vedno', 'boste', 'iv', 'vanju', 'drugega', 'nekaterega', 'osmimi', 'desetega', 'nekaterih', 'dovoliva', 'deseter', 'nikogar', 'povsod', 'čigavem', 'odprt', 'vsakršni', 'l', 'stot', 'tolika', 'midva', 'petnajsti', 'enakega', 'majhen', 'zelo', 'drugim', 'kjerkoli', 'vajinima', 'pripravljeni', 'bržčas', 'ne', 'mogla', 'stotere', 'petnajstem', 'komur', 'triinšestdesetim', 'devetnajstim', 'tja', 'brez', 'te', 'želita', 'enimi', 'osemnajstim', 'marate', 'vajinih', 'kolikšna', 'drugačno', 'čigavo', 'prvega', 'mojih', 'če', 'velika', 'takemle', 'odprti', 'petinosemdesetih', 'tristotih', 'devetnajstih', 'petemu', 'enajstima', 'nikakršnih', 'osemdeseti', 'šestemu', 'visoki', 'drug', 'trojemu', 'ničimer', 'tvojim', 'želimo', 'petindevetdesetih', 'polni', 'koliki', 'enemu', 'neka', 'tejle', 'poleg', 'devetdesete', 'vsej', 'hotele', 'pogodu', 'okoli', 'le-tistega', 'njenemu', 'istemu', 'istem', 'dvajsetemu', 'precej', 'treh', 'dvanajstem', 'njenega', 'kajne', 'že', 'onima', 'enaindvajsetim', 'petinsedemdesetih', 'zapored', 'petih', 'karkoli', 'trojnem', 'junij', 'morda', 'hočejo', 'tele', 'sedeminšestdesetim', 'le-tistimi', 'le-on', 'pot', 'težka', 'hočeta', 'čemurkoli', 'najinim', 'šestnajsti', 'želele', 'deseto', 'njihovo', 'šestnajste', 'tolikem', 'zmoreta', 'tretji', 'kakršnimi', 'sedeminšestdesetimi', 'g', 'primer', 'dvojno', 'petindvajsete', 'štiriindvajsetim', 'svoj', 'tvoji', 'dve', 'marsičem', 'še', 'marsikaterih', 'hala', 'čemerkoli', 'štirideseto', 'petinštirideseti', 'marala', 'dvema', 'ti', 'morava', 'dveh', 'dvojni', 'moral', 'dvojemu', 'tridesetim', 'zmočiti', 'pogosto', 'čemu', 'vsakega', 'ono', 'petinosemdeseta', 'deseta', 'sedemindvajsetih', 'šestindvajsetimi', 'kakšnimi', 'njenima', 'tolikšno', 'druga', 'nekakšnima', 'tisoče', 'dvakratni', 'čez', 'gor', 'zadnji', 'njegovo', 'nocoj', 'kakima', 'petinštirideset', 'vajine', 'triindvajsetem', 'petindevetdesetim', 'nasproti', 'moram', 'petstotimi', 'kratek', 'šestdesetimi', 'nobene', 'prvem', 'dvojnih', 'mara', 'neko', 'dvainšestdeset', 'petinsedemdesetimi', 'to', 'nista', 'petim', 'petindvajsetega', 'le-takšnim', 'petintrideseto', 'njeno', 'štirinajste', 'nje', 'nedavno', 'kamorkoli', 'četrt', 'zase', 'devetnajstimi', 'mogle', 'trikratnega', 'stoto', 'trikratnemu', 'devetima', 'šestdesetima', 'petinštirideseto', 'devete', 'petek', 'nanju', 'bil', 'enaindvajsetega', 'marec', 'žal', 'naproti', 'sedemstotim', 'iz', 'polna', 'tridesetem', 'naše', 'prva', 'tisočerega', 'vsem', 'vendar', 'njihovima', 'sme', 'bojo', 'zanj', 'svojima', 'nočemo', 'trikratno', 'ga.', 'le-tistih', 'osmo', 'dvaindevetdeseta', 'petdeseta', 'tisoči', 'malo', 'nihče', 'tisočemu', 'preden', 'hočemo', 'sedemnajstih', 'katerih', 'kolika', 'šesta', 'štiristotim', 'včasih', 'osemnajsta', 'temveč', 'tolikega', 'vsak', 'kakšna', 'biti', 'osmim', 'dovolita', 'vam', 'stotim', 'petinštiridesete', 'temale', 'moremo', 'trikratnim', 'njihovemu', 'dvojnemu', 'svojim', 'le-onih', 'petera', 'vašemu', 'velik', 'predvsem', 'vsakimi', 'osemdeset', 'kje', 'c', 'marsikatero', 'ali', 'dvanajsto', 'vašem', 'le-tisto', 'trojnim', 'njegovim', 'nadenj', 'kakršnimikoli', 's', 'le-takšen', 'dvoje', 'moglo', 'tvoj', 'stoteremu', 'vsakršen', 'takole', 'oziroma', 'prazno', 'zmoremo', 'dvojen', 'najsi', 'takšno', 'mi', 'devetdesetih', 'veliko', 'enajsta', 'dvainšestdesetih', 'dovoljena', 'vami', 'tristotega', 'ki', 'hoteli', 'prej', 'kolikima', 'kdaj', 'tisočeri', 'le-temu', 'marsikaterega', 'malce', 'četrtega', 'štiriindvajsete', 'seveda', 'rade', 'osemdeseta', 'čigavima', 'zato', 'kolikšnimi', 'dvajsetim', 'dvaindevetdeseti', 'gospa', 'tisočerih', 'trinajstim', 'tristotem', 'enaintrideset', 'sedemindvajseti', 'tolikšni', 'ia', 'vsaki', 'le-takemu', 'trojne', 'peterega', 'tolike', 'vred', 'trinajste', 'naša', 'triinšestdeset', 'nekemu', 'dovolijo', 'šestem', 'devet', 'vama', 'onega', 'nekakšnemu', 'najinih', 'nečem', 'tolikšnimi', 'trinajstimi', 'šestdeseta', 'tvojimi', 'lahko', 'peter', 'nek', 'česarkoli', 'eno', 'sedmima', 'desete', 'šestim', 'svoje', 'trikratni', 'njim', 'peteri', 'marsikaterima', 'takihle', 'trojen', 'smeta', 'oz.', 'devetdeseto', 'nikar', 'takšnima', 'zmogli', 'dvakratnega', 'petintrideseta', 'so', 'petima', 'nobena', 'devetdesetemu', 'najina', 'devetstotimi', 'njihovimi', 'celo', 'č', 'njihovem', 'potem', 'morajo', 'proti', 'dvaindevetdesetemu', 'šesti', 'vame', 'bi', 'nikakršnimi', 'tristo', 'našimi', 'dvaindvajsetih', 'katerimikoli', 'gotovo', 'le-tiste', 'sto', 'čigavi', 'ponedeljek', 'drugačnima', 'kogarkoli', 'trojima', 'štiridesetim', 'seboj', 'skozme', 'nekakšnem', 'enaindvajsetimi', 'dovolila', 'vsakima', 'nam', 'marali', 'visoka', 'nismo', 'drugačen', 'dvaindevetdeseto', 'trinajsta', 'srednji', 'dvajsete', 'moreva', 'toda', 'peti', 'pravi', 'le-t', 'dva', 'kot', 'devetstotim', 'tema', 'drugačna', 'neke', 'osma', 'želiva', 'osmemu', 'dvaindvajsetim', 'stotega', 'prazna', 'tisočera', 'vajinega', 'petnajste', 'zaradi', 'peterem', 'sploh', 'štiriindvajsetega', 'dovolile', 'dvojega', 'devetnajsto', 'jaz', 'pet', 'petnajstemu', 'visoke', 'petinpetdesetimi', 'čigavimi', 'najinima', 'tretjim', 'le-tistim', 'pote', 'zate', 'neradi', 'sedemindvajsete', 'moji', 'hotimo', 'kratke', 'sedemdeseta', 'čeprav', 'nikomur', 'sicer', 'kolikšen', 'enaindvajsetih', 'stotera', 'kako', 'takegale', 'saj', 'osemnajstega', 'no', 'j', 'tretjo', 'triintrideseta', 'mojemu', 'nikakršno', 'petdesetih', 'vase', 'petinosemdeseti', 'v', 'devetintridesetim', 'hoti', 'kakršnima', 'tvojem', 'petimi', 'kolik', 'ampak', 'našemu', 'jo', 'kakršne', 'nerad', 'nikakršnima', 'kom', 'tisočerimi', 'štirinajst', 'dvakratne', 'trojnima', 'petstotim', 'izven', 'želim', 'takšnimi', 'mar', 'niso', 'triintridesetimi', 'čim', 'tretja', 'trideset', 'nanje', 'kjer', 'štiriindvajseti', 'ste', 'le', 'petindvajseto', 'svojemu', 'peterim', 'triintridesetega', 'enkratnimi', 'dvanajsta', 'njima', 'dovoljene', 'vsakršnem', 'nikdar', 'pripravljena', 'drugo', 'bosta', 'nekakšno', 'peta', 'tisočimi', 'nisva', 'pravzaprav', 'smo', 'nočeva', 'osmima', 'dolg', 'nekatero', 'vsakršna', 'blizu', 'temu', 'tolikšnih', 'enkratnemu', 'morati', 'onstran', 'osmega', 'vsega', 'malone', 'dokler', 'osemnajstimi', 'sedmemu', 'drugačnimi', 'obme', 'trojem', 'štirinajstem', 'štiri', 'petinpetdesetim', 'osemdesetega', 'vrhu', 'nečemu', 'tolikšnima', 'čigava', 'šestnajstimi', 'njun', 'tolik', 'enajstih', 'takimile', 'ja', 'kateremukoli', 'kateri', 'maram', 'noben', 'triintrideseto', 'šestdesetega', 'dvajsetem', 'katerim', 'navkljub', 'sedme', 'le-ono', 'nami', 'sedemdesetem', 'četrtih', 'taki', 'običajno', 'vajin', 'daleč', 'moči', 'petnajstima', 'devetnajsta', 'ž', 'petdesete', 'šestnajstem', 'vsakršnim', 'tretjem', 'zgolj', 'sedeminpetdesetim', 'julij', 'prek', 'hočeš', 'i', 'želene', 'predse', 'tristoto', 'stoterih', 'prvo', 'komaj', 'r', 'dvakratnemu', 'le-onega', 'le-takšna', 'bova', 'želi', 'čigave', 'smete', 'slab', 'devetnajsti', 'kolikšnemu', 'vsakršnimi', 'tistima', 'želijo', 'štirinajstega', 'sama', 'koliko', 'oni', 'petintridesetega', 'našega', 'naš', 'šeststotih', 'všeč', 'takim', 'mojimi', 'dvojnega', 'zaprta', 'štiristo', 'petinsedemdesetim', 'ponje', 'enima', 'dolgi', 'hotite', 'petdesetima', 'le-takima', 'nikakršni', 'kolikšne', 'ves', 'triinšestdesetimi', 'njenih', 'želena', 'dvestotima', 'nase', 'mojim', 'enaindvajset', 'vašim', 'april', 'medtem', 'devetsto', 'prav', 'tisti', 'sedemindvajsetim', 'tretjega', 'nekimi', 'devetdeseta', 'dvajsetega', 'sedemindvajset', 'kljub', 'tvojemu', 'kakšni', 'trojim', 'čigavemu', 'petindvajseta', 'takšne', 'šestintridesetih', 'moč', 'moreta', 'sedemdeseti', 'devetnajstima', 'onimi', 'dvestoto', 'sobota', 'pač', 'bili', 'le-tistemu', 'nekem', 'smel', 'petinštiridesetih', 'skoz', 'štiridesetemu', 'do', 'za', 'dovoliš', 'sedemindvajseta', 'zlasti', 'maj', 'enkratno', 'dvakraten', 'šestega', 'tem', 'mene', 'onkraj', 'dvakratnih', 'tristotim', 'sedemnajstega', 'marale', 'nič', 'kadar', 'osemindevetdeset', 'morala', 'eni', 'peterih', 'niste', 'trije', 'šestnajstemu', 'devetstotih', 'tristoti', 'tisočerim', 'petnajstih', 'idr.', 'u', 'tolikšne', 'tremi', 'oseminštiridesetimi', 'sedemindvajsetimi', 'njeni', 'stotih', 'triindvajseti', 'petindvajsetima', 'petindvajsetim', 'enajstega', 'enaka', 'triintridesetem', 'trojnih', 'osmem', 'tisočeremu', 'sedemdesetimi', 'osemdesete', 'drugačnem', 'nekatera', 'zanje', 'le-oniti', 'kolikim', 'samo', 'ko', 'tvojo', 'želen', 'torek', 'hotela', 'sedemnajstem', 'štiriindvajsetemu', 'sedemdesetega', 'trideseto', 'štirimi', 'devetdesetim', 'štirinajstim', 'medme', 'triindvajsete', 'sedaj', 'ponjo', 'najino', 'danes', 'edinole', 'triindvajsetimi', 'petstotih', 'dvojnim', 'le-tej', 'kakršnekoli', 'želelo', 'vsa', 'dvaindevetdesetim', 'tega', 'dvaindevetdesetima', 'triintrideseti', 'kdo', 'trikratna', 'dovolili', 'deveti', 'čigar', 'mojima', 'zaprti', 'npr.', 'kakšnim', 'naj', 'njihovi', 'zakaj', 'najinemu', 'katera', 'dvanajstega', 'dvojih', 'osemnajsto', 'nekakšni', 'nji', 'e', 'nadnje', 'stoterima', 'šestnajstim', 'petintridesetimi', 'iia', 'dol', 'dovolimo', 'štirinajstimi', 'šestdesetemu', 'osemdeseto', 'petintrideseti', 'enaindvajsetemu', 'troje', 'sedemdesetih', 'osemnajsti', 'poln', 'vašima', 'enih', 'petnajst', 'tistem', 'visok', 'četrtimi', 'boš', 'marsikaj', 'le-ona', 'le-temi', 'trinajst', 'dvajsetih', 'nisem', 'vsakem', 'petindevetdeset', 'kakšnega', 'ker', 'toliki', 'ničemur', 'dvajsetima', 'zoper', 'kajpada', 'želeni', 'devetintridesetimi', 'vate', 'nekakima', 'zmorejo', 'njene', 'morejo', 'tistemu', 'dvakratna', 'leto', 'nekaj', 'nekomu', 'štirim', 'zunaj', 'enaki', 'š', 'dvainšestdesetimi', 'dvestotem', 'osemnajstih', 'ix', 'morate', 'enim', 'marsičemu', 'takšnih', 'dvoj', 'enajstemu', 'njen', 'domala', 'petindvajsetem', 'tistega', 'takšen', 'vanj', 'onemu', 'kaj', 'nikakršne', 'nekakega', 'desetih', 'njunima', 'osemdesetem', 'dvojne', 'pribl.', 'dvaindevetdesetem', 'tristotemu', 'stote', 'onedve', 'štiridesetem', 'vštric', 'vajinemu', 'temuintem', 'marsikateremu', 'onadva', 'tistih', 'odprta', 'petinpetdesetih', 'kakšnima', 'njunih', 'marsičesa', 'tehle', 'prvim', 'petero', 'nekdo', 'marveč', 'zdaj', 'štiriindvajsetem', 'deveta', 'tridesetima', 'marajo', 'dvoji', 'morebiti', 'vsakršne', 'pri', 'včeraj', 'tolikim', 'triindvajset', 'koderkoli', 'marava', 'istega', 'tistim', 'druge', 'šestnajstega', 'smele', 'enak', 'istima', 'kakršnihkoli', 'hotelo', 'enajsti', 'meni', 'pravo', 'triintrideset', 'vajini', 'ona', 'oseminštiridesetim', 'štirinajsto', 'medse', 'zmorete', 'september', 'dovoljen', 'petdeset', 'le-ti', 'drugi', 'avgust', 'njihov', 'prednji', 'halo', 'le-teh', 'dvanajste', 'dvaindevetdeset', 'onim', 'enaindvajsetima', 'šestdesetem', 'petinosemdesetem', 'kolikšnim', 'petindvajsetimi', 'štirinajsti', 'vsakršnega', 'dvanajstim', 'oboje', 'dvajsetimi', 'nočeš', 'sam', 'ji', 'namreč', 'šestdeset', 'tretje', 'šestima', 'midve', 'takima', 'drugačnemu', 'petnajsto', 'stotemu', 'kakih', 'tridesetega', 'desetero', 'nisi', 'trinajsti', 'sedemdesetim', 'mimo', 'sedeminšestdesetih', 'enake', 'naju', 'enkrat', 'trojnimi', 'le-takim', 'način', 'petindevetdesetimi', 'sedmi', 'četrtemu', 'ponavadi', 'zmorem', 'le-takšnih', 'deseterih', 'vajinem', 'bodite', 'b', 'petnajstim', 'petinosemdesetimi', 'spričo', 'zaprto', 'devetimi', 'mora', 'komerkoli', 'petnajstega', 'moralo', 'zanjo', 'štiridesete', 'via', 'petinštiridesetem', 'želel', 'nekakšnimi', 'deseteremu', 'nekakšen', 'petinosemdeset', 'dvanajstih', 'čeznje', 'dan', 'nemara', 'osemindevetdesetimi', 'drugih', 'bile', 'četrto', 'dovoljeni', 'bilo', 'tak', 'marsikom', 'na', 'kakšnih', 'ta', 'nobenim', 'dovoli', 'onidve', 'devetnajstem', 'vsakim', 'njih', 'deseteri', 'enakimi', 'trikratnem', 'stoterim', 'takimale', 'četrtek', 'tolikima', 'petnajsta', 'dvestotimi', 'le-toliko', 'dovoljeno', 'četrte', 'šeste', 'dno', 'navzlic', 'prazen', 'sedemdeseto', 'največ', 'ju', 'sedeminpetdesetimi', 'njenimi', 'vaju', 'enaindvajseta', 'vas', 'dvestotim', 'njuni', 'bodi', 'prve', 'm', 'res', 'vaš', 'koli', 'marsikomu', 'tolikšnem', 'ponovno', 'podnjo', 'dvakratnim', 'petindvajsetih', 'petdesetemu', 'kolikor', 'katerega', 'marajte', 'koder', 'en', 'krog', 'k', 'desetere', 'le-tistem', 'le-take', 'tristote', 'isti', 'vsakemu', 'njegov', 'desetem', 'bodimo', 'naokoli', 'katero', 'enako', 'petintridesetih', 'osemdesetih', 'lepi', 'štirinajstemu', 'verjetno', 'šestindvajsetim', 'marati', 'kakršnemu', 'nekateremu', 'dobra', 'oseminštiridesetih', 'zadosti', 'tisočima', 'drugačnih', 'mojem', 'tole', 'kakimi', 'bodo', 'drugačne', 'kakršno', 'morata', 'ii', 'četrtima', 'šestdeseti', 'sedmih', 'smeli', 'zaprt', 'petinosemdesete', 'mnogo', 'devetintrideset', 'tisočero', 'le-takih', 'komurkoli', 'vsakomur', 'devetnajst', 'sedemnajstimi', 'sedemindvajsetega', 'sedemindvajsetima', 'nadvse', 'noče', 'dvojnima', 'nikakršnega', 'naših', 'dvaindvajsetimi', 'trojnemu', 'vsakršnemu', 'tebe', 'njuna', 'svojimi', 'ni', 'tolikemu', 'torej', 'kakega', 'datum', 'devetih', 'sedemdesetima', 'prvima', 'štiriindvajset', 'petinštiridesetim', 'tristotima', 'zmore', 'triinšestdesetih', 'le-takšne', 'sva', 'menda', 'le-tistima', 'tolikšna', 'nikakršnim', 'enaindvajsete', 'osemnajste', 'vsakomer', 'šestindvajset', 'nobenima', 'teh', 'kolikega', 'kolikem', 'maramo', 'takih', 'četrta', 'dvajseti', 'takšnim', 'o', 'nekatere', 'hočete', 'lahka', 'kakršenkoli', 'menoj', 'obema', 'sedemnajsto', 'deseterem', 'nate', 'sedemnajstemu', 'kdor', 'zmoreš', 'nikakršnemu', 'nekakemu', 'takšnem', 'svoji', 'sedemdeset', 'dvojimi', 'teboj', 'sedemnajst', 'jutri', 'le-taki', 'zmoči', 'tebi', 'istimi', 'marsikoga', 'vsakršnih', 'štirih', 'on', 'nobenega', 'narobe', 'osemdesetima', 'tabo', 'trojimi', 'tisočem', 'kolike', 'našim', 'enkratnima', 'petindvajseti', 'sabo', 'marsikatera', 'temi', 'je', 'vnovič', 'osemnajst', 'nekje', 'desetera', 'stoterimi', 'nekom', 'enkratni', 'vsemu', 'kolikšnega', 'le-takšnega', 'njimi', 'njune', 'povrhu', 'triindvajsetega', 'sedeminpetdeset', 'njihovega', 'skoznje', 'petinštiridesetega', 'kaki', 'vašimi', 'le-teti', 'moja', 'stoteri', 'okrog', 'šestdesetim', 'kak', 'le-te', 'kakršnikoli', 'nekakšnim', 'dovolilo', 'dovoliti', 'čemur', 'ničesar', 'štirideseti', 'le-ta', 'le-takega', 'zraven', 'jim', 'njunimi', 'stotima', 'marajmo', 'štiriindvajsetih', 'nočeta', 'dvojem', 'takimi', 'petdesetega', 'hoteti', 'tristota', 'ponj', 'takšni', 'december', 'baje', 'prbl.', 'le-onima', 'd', 'tam', 'njemu', 'trikratnima', 'majhna', 'devetim', 'petdesetimi', 'tu', 'osemdesetimi', 'njihova', 'p', 'petnajstimi', 'dovolj', 'šeststo', 'šestdesetih', 'enaindvajseti', 'tudi', 'petsto', 'pred', 'sebi', 'lahek', 'njega', 'maral', 'našo', 'takšnemu', 'enakih', 'nikakršna', 'nekakim', 'želeli', 'težek', 'le-tega', 'sedemnajsta', 'prvemu', 'le-to', 'najmanj', 'petintrideset', 'reč', 'sedemstotimi', 'enkratna', 'moreš', 'med', 'trinajstima', 'nanjo', 'mednje', 'kakršen', 'kajpak', 'četrti', 'nekakem', 'tri', 'petinštiridesetima', 'šestintridesetimi', 'vanjo', 'iva', 'petem', 'enaintridesetimi', 'petinosemdesetima', 'petdesetim', 'hoče', 'ničemer', 'peto', 'celi', 'oba', 'drugima', 'katere', 'vedeti', 'štirinajsta', 'smelo', 'čem', 'enkratnega', 'kateremu', 'kar', 'tridesetih', 'dvoja', 'kakšnem', 'enkraten', 'osemindevetdesetih', 'dvajseta', 'vsema', 'čigavih', 'a', 'pete', 'tvojima', 'štirje', 'moje', 'vsaj', 'zares', 'vsakogar', 'vašega', 'nobeno', 'njej', 'me', 'česar', 'lahke', 'ena', 'meja', 'vajino', 'štiridesetih', 'le-tema', 'kolikemu', 'sedemnajste', 'stota', 'x', 'sedemstotih', 'nekakšne', 'smela', 'stoterega', 'trojni', 'tvoje', 'trikratnih', 'nikoli', 'dobri', 'šestnajsta', 'desetimi', 'čigavega', 'osme', 'bodiva', 'devetnajstega', 'želite', 'nekaterimi', 'tisočere', 'čigav', 'nekaterim', 'iii', 'mej', 'težak', 'štirideset', 'enajsto', 'hotita', 'petintridesetemu', 'tako', 'nekega', 'tolikšen', 'kolikšnima', 'dvesto', 'dvaindevetdesetimi', 'radi', 'devetdesetega', 'nočete', 'enaintridesetim', 'kakršnimkoli', 't', 'stoter', 'sedemnajsti', 'tolikšnega', 'majhni', 'g.', 'nekim', 'smet', 'spet', 'petere', 'dvestoti', 'srednja', 'deseterim', 'bom', 'petintridesetem', 'tale', 'šele', 'često', 'triinpetdeset', 'nečim', 'sem', 'njihovim', 'takoj', 'nanj', 'sedeminpetdesetih', 'sredi', 'tile', 'preprosto', 'težko', 'kam', 'dvaindevetdesete', 'devetem', 'maraš', 'najbrž', 'dvakratnem', 'petinštiridesetimi', 'katerakoli', 'kakršnegakoli', 'najini', 'nobenemu', 'mogli', 'vpričo', 'kratka', 'šestintrideset', 'ga', 'devetnajste', 'marsikdo', 'ter', 'štiriindvajseto', 'šest', 'lep', 'trinajstega', 'njunega', 'troji', 'enajstem', 'temle', 'nobeni', 'skozte', 'triintridesetim', 'tisočer', 'osmih', 'sedemsto', 'kateremkoli', 'trem', 'dvaindvajset', 'stotimi']



def read_csv_with_embeddings(file_path, embedding_column="embedding"):

    def convert_embedding(embedding_str):
        return np.array([float(x) for x in embedding_str.strip('[]').split()])

    df = pd.read_csv(file_path)
    
    df[embedding_column] = df[embedding_column].apply(convert_embedding)
    
    return df 

data =  [ {"category": "sport", "occurrences": 170, "mae": 40.718130111694336, "best_epoch": 49 , "parameters": {'n_layers': 2, 'n_units_l0': 87, 'n_units_l1': 111, 'dropout_l0': 0.1710113526306438, 'dropout_l1': 0.15024468145809303, 'batch_size': 128, 'weight_decay': 0.0002585967293988982}}, {"category": "svet", "occurrences": 122, "mae": 28.775415420532227, "best_epoch": 41, "parameters": {'n_layers': 3, 'n_units_l0': 131, 'n_units_l1': 118, 'n_units_l2': 116, 'dropout_l0': 0.1823609040810445, 'dropout_l1': 0.0738109845763633, 'dropout_l2': 0.40432778637003436, 'batch_size': 122, 'weight_decay': 0.001553718760285118}}, {"category": "kultura", "occurrences": 88, "mae": 4.021245002746582, "best_epoch": 53 , "parameters": {'n_layers': 4, 'n_units_l0': 58, 'n_units_l1': 103, 'n_units_l2': 120, 'n_units_l3': 22, 'dropout_l0': 0.377071002952924, 'dropout_l1': 0.4945196557419954, 'dropout_l2': 0.376097501954881, 'dropout_l3': 0.14597844409568736, 'batch_size': 88, 'weight_decay': 0.01809908522346439}}, {"category": "zabava-in-slog", "occurrences": 84, "mae": 29.93294906616211, "best_epoch": 53, "parameters": {'n_layers': 2, 'n_units_l0': 123, 'n_units_l1': 93, 'dropout_l0': 0.13964574705169086, 'dropout_l1': 0.0794452196130308, 'batch_size': 84, 'weight_decay': 9.189879886915706e-05}}, {"category": "slovenija", "occurrences": 60, "mae": 30.65727996826172, "best_epoch": 64 , "parameters": {'n_layers': 4, 'n_units_l0': 128, 'n_units_l1': 30, 'n_units_l2': 24, 'n_units_l3': 40, 'dropout_l0': 0.44004611168727126, 'dropout_l1': 0.056386642245860645, 'dropout_l2': 0.21518495150248568, 'dropout_l3': 0.47928164883404695, 'batch_size': 60, 'weight_decay': 0.0022667201477713107}}, {"category": "gospodarstvo", "occurrences": 20, "mae": 29.685483932495117, "best_epoch": 34 , "parameters": {'n_layers': 2, 'n_units_l0': 64, 'n_units_l1': 140, 'dropout_l0': 0.293803050205372, 'dropout_l1': 0.2439297693199976, 'batch_size': 20, 'weight_decay': 0.0012636891460772455}}, {"category": "crna-kronika", "occurrences": 10, "mae": 12.85865306854248, "best_epoch": 18 , "parameters": {'n_layers': 3, 'n_units_l0': 85, 'n_units_l1': 70, 'n_units_l2': 102, 'dropout_l0': 0.10859050686790658, 'dropout_l1': 0.4400337787043451, 'dropout_l2': 0.021474695853379674, 'batch_size': 10, 'weight_decay': 0.006903939798477992}}, {"category": "okolje", "occurrences": 9, "mae": 20.168025970458984, "best_epoch": 66 , "parameters": {'n_layers': 3, 'n_units_l0': 52, 'n_units_l1': 20, 'n_units_l2': 66, 'dropout_l0': 0.33050229638658646, 'dropout_l1': 0.28965133216236777, 'dropout_l2': 0.38318445625094383, 'batch_size': 9, 'weight_decay': 0.00343777386221657}}, {"category": "znanost-in-tehnologija", "occurrences": 4, "mae": 1.3390145301818848, "best_epoch": 39, "parameters": {'n_layers': 2, 'n_units_l0': 27, 'n_units_l1': 119, 'dropout_l0': 0.217381221875468, 'dropout_l1': 0.20181479389005275, 'batch_size': 4, 'weight_decay': 0.02336121763946577}}, {"category": "stevilke", "occurrences": 1, "mae": 22.957040786743164, "best_epoch": 31 , "parameters": {'n_layers': 4, 'n_units_l0': 58, 'n_units_l1': 103, 'n_units_l2': 120, 'n_units_l3': 22, 'dropout_l0': 0.377071002952924, 'dropout_l1': 0.4945196557419954, 'dropout_l2': 0.376097501954881, 'dropout_l3': 0.14597844409568736, 'batch_size': 1, 'weight_decay': 0.01809908522346439}} ]


##################################################
##################################################
#PREPROCESS DATASET - USES GPU
#dla, na gpu

# Function to process text and extract entities, lemmatized text, and count entities
def process_text(doc):
    print("+")
    entities = list({(ent.text, ent.label_) for ent in doc.ents})
    
    # Lemmatize and remove non-alphabetic characters and stopwords
    lemmatized_text = " ".join([token.lemma_ for token in doc if token.is_alpha and token.text.lower() not in slo_stopwords])
    
    # Remove punctuation and convert to lowercase
    lemmatized_text = re.sub(r'[^\w\s]', '', lemmatized_text).lower().strip()
    
    # Remove extra whitespace
    lemmatized_text = re.sub(r'\s+', ' ', lemmatized_text)
    
    # Count entity types
    ne_loc_cnt = sum(1 for _, label in entities if label in {'GPE', 'LOC'})
    ne_per_cnt = sum(1 for _, label in entities if label == 'PER')
    ne_org_cnt = sum(1 for _, label in entities if label == 'ORG')
    ne_misc_cnt = sum(1 for _, label in entities if label == 'MISC')
    print("+")

    return entities, lemmatized_text, ne_loc_cnt, ne_per_cnt, ne_org_cnt, ne_misc_cnt



# Define one hour in seconds
one_hour_in_seconds = 3600

# Function to count articles in one-hour interval for each category
def count_articles_in_interval(timestamp, timestamps, topics, category, interval=one_hour_in_seconds):
    start_interval = timestamp - interval
    end_interval = timestamp + interval
    return np.sum((timestamps >= start_interval) & (timestamps <= end_interval) & (topics == category))



def preprocess(df,output_name,):
    if output_name == "train_df.csv":
        df = df.dropna(subset=['topics'])
    ##DROP CATEGORY if exists
    if 'category' in df.columns:
        df = df.drop(columns=['category'])
    
    # Check if CUDA is available and set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Print GPU and CUDA details if available
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("Using CPU")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("EMBEDDIA/sloberta")
    model = AutoModel.from_pretrained("EMBEDDIA/sloberta").to(device)
    #print model size
    def combine_columns(row):
        title = row['title']
        lead = row['lead']
        paragraphs = " ".join(row['paragraphs']) if isinstance(row['paragraphs'], list) else row['paragraphs']
        keywords = ", ".join(row['keywords']) if isinstance(row['keywords'], list) else row['keywords']
        gpt_keywords = ", ".join(row['gpt_keywords']) if isinstance(row['gpt_keywords'], list) else row['gpt_keywords']
        
        combined_text = f"Title: {title}\n\nLead: {lead}\n\nContent: {paragraphs}\n\nKeywords: {keywords}\n\nGPT Keywords: {gpt_keywords}"
        return combined_text
    
    def category_calculation(df):
    # Load the DataFrame
    
        # Function to extract categories from the URL
        def extract_category(url):
            # Split the URL on "/"
            parts = url.split('/')
            # Find the index for ".si"
            index = parts.index('www.rtvslo.si') + 1
            # Return the next two parts joined by "/"
            return '/'.join(parts[index:index+2])

        # Apply the function to the 'url' column
        df['category_new'] = df['url'].apply(extract_category)

        # Count the unique values in the 'category' column
        category_counts = df['category_new'].value_counts()

        # Calculate the 0.5% threshold of the total dataset size
        threshold = len(df) * 0.005

        # Filter categories that appear more than the threshold
        significant_categories = category_counts[category_counts > threshold].index.tolist()

        # Define a function to adjust category based on significance
        def adjust_category(category):
            # Check if the category is significant
            if category not in significant_categories:
                # If not, return the higher-level hierarchy
                return category.split('/')[0]
            else:
                # If it is, return the category as is
                return category

        # Apply the function to adjust categories
        df['category_new'] = df['category_new'].apply(adjust_category)
        
        
    
    def combine_columns_lead_title(row):
        title = row['title']
        lead = row['lead']  
        keywords = ", ".join(row['keywords']) if isinstance(row['keywords'], list) else row['keywords']
        gpt_keywords = ", ".join(row['gpt_keywords']) if isinstance(row['gpt_keywords'], list) else row['gpt_keywords']

        combined_text = f"{title} {lead} {keywords} {gpt_keywords}"
        return combined_text
    
    
    # Function to split text into chunks
    def split_into_chunks(text, chunk_size=512, overlap=50):
        tokens = tokenizer.tokenize(text)
        chunks = []
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk = tokens[i:i + chunk_size]
            chunks.append(tokenizer.convert_tokens_to_string(chunk))
        return chunks

    # Function to compute embeddings
    def compute_embedding(text):
        chunks = split_into_chunks(text)
        chunk_embeddings = []
        for chunk in chunks:
            inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            chunk_embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy())
        print("-")
        return np.mean(chunk_embeddings, axis=0)




    df['combined_text'] = df.apply(combine_columns, axis=1)
    df['combined_text_lead_title'] = df.apply(combine_columns_lead_title, axis=1)

    time_start = time.time()
    # Compute embeddings for the combined text
    df['embedding'] = df['combined_text'].apply(compute_embedding)
    time_end = time.time()
    print("Time taken to compute embeddings:", time_end - time_start, "seconds")
    
##TIME TO SIN COS
        # Convert to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Convert to Unix timestamp
    df['unix_timestamp'] = df['date'].astype(int) / 10**9

    # Normalize by subtracting the minimum value
    min_timestamp = df['unix_timestamp'].min()
    df['unix_timestamp'] = df['unix_timestamp'] - min_timestamp

    # Define the periods (in seconds) for different cycles
    seconds_in_day = 24 * 60 * 60
    seconds_in_week = 7 * seconds_in_day
    seconds_in_month = 30 * seconds_in_day  # Approximation
    seconds_in_quarter = 3 * seconds_in_month
    seconds_in_half_year = 6 * seconds_in_month
    seconds_in_year = 12 * seconds_in_month

    # Apply sine and cosine transformations for different cycles
    # Daily cycle
    df['daily_sin'] = np.sin(2 * np.pi * df['unix_timestamp'] / seconds_in_day)
    df['daily_cos'] = np.cos(2 * np.pi * df['unix_timestamp'] / seconds_in_day)

    # Three-day cycle
    df['three_day_sin'] = np.sin(2 * np.pi * df['unix_timestamp'] / (3 * seconds_in_day))
    df['three_day_cos'] = np.cos(2 * np.pi * df['unix_timestamp'] / (3 * seconds_in_day))

    # Weekly cycle
    df['weekly_sin'] = np.sin(2 * np.pi * df['unix_timestamp'] / seconds_in_week)
    df['weekly_cos'] = np.cos(2 * np.pi * df['unix_timestamp'] / seconds_in_week)

    # Monthly cycle
    df['monthly_sin'] = np.sin(2 * np.pi * df['unix_timestamp'] / seconds_in_month)
    df['monthly_cos'] = np.cos(2 * np.pi * df['unix_timestamp'] / seconds_in_month)

    # Quarterly cycle
    df['quarterly_sin'] = np.sin(2 * np.pi * df['unix_timestamp'] / seconds_in_quarter)
    df['quarterly_cos'] = np.cos(2 * np.pi * df['unix_timestamp'] / seconds_in_quarter)

    # Half-year cycle
    df['half_yearly_sin'] = np.sin(2 * np.pi * df['unix_timestamp'] / seconds_in_half_year)
    df['half_yearly_cos'] = np.cos(2 * np.pi * df['unix_timestamp'] / seconds_in_half_year)

##NUMBER OF FIGURES
    df['figure_count'] = df['figures'].apply(len)
    
##NUMBER OF PARAGRAPHS
    df['paragraph_count'] = df['paragraphs'].apply(len)

    
##ARTICLE LENGTH
    df['article_word_count'] = df['combined_text'].apply(lambda text: len(text.split()))
    
##TITLE LENGTH
    df['title_word_count'] = df['title'].apply(lambda text: len(text.split()))
#DOES TITLE CONTAIN ! OR ?  Create features indicating the presence of '?' and '!' in the title
    df['artquestion'] = df['title'].apply(lambda x: '?' in x)
    df['artexclaim'] = df['title'].apply(lambda x: '!' in x)
    
#ARTICLES IN ONE HOUR    

    try:
        # Try to open the file and load the list
        with open("unique_col.json", 'r') as file:
            unique_topics = json.load(file)
    except FileNotFoundError:
        # If the file does not exist, create the list and save it
        unique_topics = list(df['topics'].unique())
        with open("unique_col.json", 'w') as file:
            json.dump(unique_topics, file)
        
    
    # Iterate over each unique topic and calculate the count in the one-hour interval
    for topic in unique_topics:
        df[f'{topic}_count_in_one_hour'] = df.apply(
            lambda row: count_articles_in_interval(row['unix_timestamp'], df['unix_timestamp'], df['topics'], topic),
            axis=1
        )

    timestart = time.time()
# Apply functions to count different types of entities
# Process the text and save entities
# Apply the NLP pipeline and process the text
    df['doc'] = df['combined_text_lead_title'].apply(nlp)
    df[['entities', 'lemmatized_lead_title', 'ne_loc_cnt', 'ne_per_cnt', 'ne_org_cnt', 'ne_misc_cnt']] = df['doc'].apply(lambda doc: pd.Series(process_text(doc)))


    """     df['doc'] = df['combined_text_lead_title'].apply(nlp)
    df['entities'] = df['doc'].apply(lambda doc: list({(ent.text, ent.label_) for ent in doc.ents}))
    df['ne_loc_cnt'] = df['entities'].apply(count_location_entities_from_list)
    df['ne_per_cnt'] = df['entities'].apply(count_person_entities_from_list)
    df['ne_org_cnt'] = df['entities'].apply(count_org_entities_from_list)  """

    timeend = time.time()
    print("Time taken to count entities-Spacy bert task:", timeend - timestart, "seconds")

    #CATEGORY NEW CALCULATION
    category_calculation(df)


    #THE SAME df BUT SORT COLUMNS ACCORDING TO THE ALPHABET
    df = df.reindex(sorted(df.columns), axis=1)
    df.to_csv(output_name, index=False) 
    return df















def unpack_embeddings(df, drop_original=True):
    # Assuming each embedding has the same dimension
    embedding_dim = len(df.iloc[0]['embedding'])
    
    # Create a new column for each dimension of the embedding
    for i in range(embedding_dim):
        df[f'emb_{i}'] = df['embedding'].apply(lambda x: x[i])

    # Optionally drop the 'embedding' column
    if drop_original:
        df.drop('embedding', axis=1, inplace=True)

    return df

columns_to_drop = ['url', 'authors', 'date', 'title', 'paragraphs', 'figures', 'lead',
                   'topics', 'keywords', 'gpt_keywords', 'id', 'combined_text',
                   'combined_text_lead_title', 'unix_timestamp',
                   'doc', 'entities', 'lemmatized_lead_title', 'ne_misc_cnt']

globalBestMae=100000.0
def getPredictions(df,df_test,dictionary):
    # Load the DataFrame

    
    ##################################################
    ##################################################  
         # Initialize the OneHotEncoder
    # Concatenate 'category_new' columns from both DataFrames for fitting the encoder
    combined_categories = pd.concat([df[['category_new']], df_test[['category_new']]])

    # Initialize the OneHotEncoder
    encoder = OneHotEncoder(sparse=False)

    # Fit the encoder on the combined data
    encoder.fit(combined_categories)

    # Transform the 'category_new' column for both DataFrames
    encoded_df = pd.DataFrame(encoder.transform(df[['category_new']]), columns=encoder.get_feature_names_out(['category_new']))
    encoded_df_test = pd.DataFrame(encoder.transform(df_test[['category_new']]), columns=encoder.get_feature_names_out(['category_new']))

    # Drop the original 'category_new' column and concatenate the new one-hot encoded columns
    df = df.drop('category_new', axis=1).reset_index(drop=True)
    df_test = df_test.drop('category_new', axis=1).reset_index(drop=True)

    df = pd.concat([df, encoded_df], axis=1)
    df_test = pd.concat([df_test, encoded_df_test], axis=1)  

    #####################################
    #####################################

    #df = pd.read_csv("X_train_sport.csv")
    df.drop(columns_to_drop, axis=1, inplace=True)
    X_train = df.drop('n_comments', axis=1).values
    y_train = df['n_comments'].values

    y_train=np.sqrt(y_train)

    df_test.drop(columns_to_drop, axis=1, inplace=True)
    X_test = df_test.drop('n_comments', axis=1).values
    y_test = df_test['n_comments'].values

    # Standard scaling of features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_val=X_test
    y_val=y_test


    # Converting data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
 
    global globalBestMae
    globalBestMae=100000.0


    def initialize_weights_xavier(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)






# Define the objective function for Optuna
    INITIAL_SEED = 1
    random.seed(INITIAL_SEED)
    np.random.seed(INITIAL_SEED)
    torch.manual_seed(INITIAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(INITIAL_SEED)
        torch.cuda.manual_seed_all(INITIAL_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parameters = dictionary["parameters"]

    n_layers = parameters["n_layers"]
    layers = [parameters[f"n_units_l{i}"] for i in range(n_layers)]
    dropout_rates = [parameters[f"dropout_l{i}"] for i in range(n_layers)]
    weight_decay = parameters["weight_decay"]
    batch_size = parameters["batch_size"]
    epochs = dictionary["best_epoch"]
    # Set a fixed seed value for initialization

    # Building the neural network model
    model = nn.Sequential()
    input_size = X_train.shape[1]
    for i in range(n_layers):
        output_size = layers[i]
        model.add_module(f"linear{i}", nn.Linear(input_size, output_size))
        model.add_module(f"relu{i}", nn.ReLU())
        model.add_module(f"dropout{i}", nn.Dropout(dropout_rates[i]))
        input_size = output_size  # update input size for the next layer
    model.add_module("output", nn.Linear(layers[-1], 1))
    
    model.apply(initialize_weights_xavier)

    # Setup DataLoaders with dynamic batch size
    g = torch.Generator()
    g.manual_seed(INITIAL_SEED)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)


    # Moving model to GPU if available
    if torch.cuda.is_available():
        model.cuda()
    else:
        print("Using CPU")

    # Setup optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=weight_decay)
    criterion = nn.L1Loss()



    # Training loop
    for epoch in range(epochs+1):  # Maximum number of epochs
        model.train()
        for batch_x, batch_y in train_loader:
            if torch.cuda.is_available():
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
        # Validation loop
    predictions = []
    model.eval()
    val_mae = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            if torch.cuda.is_available():
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            output = model(batch_x)
            output=output**2
            predictions.extend(output.cpu().numpy().flatten())
            #set negative predictions to 0
            output[output<0]=0
            val_mae += criterion(output, batch_y).item()
    val_mae /= len(val_loader)
    print(f"Validation MAE: {val_mae}")

    return predictions












def read_json(data_path: str) -> list:
    with gzip.open(data_path, 'rt', encoding='utf-8') as f:
        return json.load(f)

class RTVSlo:

    def __init__(self):
        pass

    def fit(self, train_data: list):
        pass        
        
    def predict(self, test_data: list) -> np.array:
        df=read_csv_with_embeddings("SAVEtrain_df.csv")
        df=unpack_embeddings(df)
        df_test=read_csv_with_embeddings("SAVEtest_df.csv")
        df_test=unpack_embeddings(df_test)
        topics=['sport','svet','kultura','zabava-in-slog','slovenija','gospodarstvo','crna-kronika','okolje','znanost-in-tehnologija','stevilke']
        combined_predictions = [None] * len(df_test)    
        for topic in topics:
        # Select only rows with the current topic in df and df_test
        df_topic = df[df['topics'] == topic]
        df_test_topic = df_test[df_test['topics'] == topic]
        
        # Remember the indices of the selected rows in df_test
        indices = df_test[df_test['topics'] == topic].index.tolist()
        
        # Select the corresponding dictionary from data
        dictionary = next(item for item in data if item['category'] == topic)
        print(dictionary)
        # Call getPredictions
        pred = getPredictions(df_topic, df_test_topic, dictionary)
        print(pred)
            # Recombine predictions using indices
        for idx, prediction in zip(indices, pred):
            combined_predictions[idx] = prediction

    # Print the combined predictions for the whole test set
    with open("predictions.txt", "w") as file:
        for prediction in combined_predictions:
            file.write(f"{prediction}\n")   
                
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_data_path', type=str)
    parser.add_argument('test_data_path', type=str)
    args = parser.parse_args()

    #if there is train?df.csv and test_df.csv

    if os.path.exists("train_df.csv") and os.path.exists("test_df.csv"):
        pass
    else:
        train_data = read_json(args.train_data_path)
        test_data = read_json(args.test_data_path)
        
        train_df=pd.read_json(args.train_data_path)
        test_df=pd.read_json(args.test_data_path)
        preprocess(train_df,"train_df.csv")
        preprocess(test_df,"test_df.csv")

    rtv = RTVSlo()
    rtv.fit(train_data)
    predictions = rtv.predict(test_data)

    if os.path.exists('predictions.txt'):
        os.remove('predictions.txt')

    np.savetxt('predictions.txt', predictions)

if __name__ == '__main__':
    main()




