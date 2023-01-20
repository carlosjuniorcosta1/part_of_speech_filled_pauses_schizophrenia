# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 20:06:45 2022

@author: Usuario
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
import re
import xml.etree.ElementTree as ET
from collections import Counter
import pickle
import nltk
from nltk import word_tokenize
import numpy as np
import os
from string import punctuation
nltk.download('punkt')
import xml.etree.ElementTree as ET



def limpa_coral(texto):
    import re 
    texto = re.sub(r'(\*\w{3}\:)?(\[\d+\])?|\[?\/\d+\]?|\+|/|/{2}|=?i?\s?\-?\w{3}_?r?s?n?=\s?\$?', '', texto)
    texto = texto.replace('hhh', '').replace('yyyy', '')\
    .replace('yyy', '').replace('xxx', '').replace('<', '')\
    .replace('>','').replace('?', '').replace('=', '')
    
    texto = re.sub(r'&\w+', '', texto)
    texto = re.sub(r'\s+', ' ', texto)
    
    return texto   

def limpa_coral_sobreposicao(texto):
    import re 
    texto = re.sub(r'(\*\w{3}\:)?(\[\d+\])?|\[?\/\d+\]?|\+|/|/{2}|=?i?\s?\-?\w{3}_?r?s?n?=\s?\$?', '', texto)
    texto = texto.replace('hhh', '').replace('yyyy', '').replace('yyy', '').replace('xxx', '')\
    .replace('?', '').replace('=', '')
    
    texto = re.sub(r'&\w+', '', texto)
    texto = re.sub(r'\s+', ' ', texto)
    
    return texto


def normaliza_coral(texto):
    import re 
    texto = re.sub(r'(\*\w{3}\:)?(\[\d+\])?|\[?\/\d+\]?|\+|/|/{2}|=?i?\s?\-?\w{3}_?r?s?n?=\s?\$?', '', texto)
    texto = texto.replace('hhh', '').replace('yyyy', '')\
    .replace('yyy', '').replace('xxx', '').replace('<', '')\
    .replace('>','').replace('?', '').replace('=', '')
    texto = re.sub('&\w+', '', texto)
    texto = re.sub(r'&\w+', '', texto)
    texto = re.sub(r'\s+', ' ', texto)
    texto = texto.replace("'", "’")
    texto = texto.strip()
    formas_conv ="""
    ni (em), a’ (olha), acabamo (acabamos), achamo (achamos), agradecemo (agradecemos), a’ lá (olha), a’ (olha), a’ (olha), aprendemo (aprendemos), arrumamo (arrumamos), assinávamo (assinávamos), atravessamo (atravessamos), avi (vi), avinha (vinha), bebemo (bebemos), beijamo (beijamos), botemo (botamos), chegamo (chegamos), cheguemo (chegamos), choramo (choramos), colocamo (colocamos), começamo (começamos), comemo (comemos), comemoramo (comemoramos), compramo (compramos), conhecemo (conhecemos), conseguimo (conseguimos), contamo (contamos), conversamo (conversamos), corremo (corremos), cortamo (cortamos), deixamo (deixamos), descansamo (descansamos), descemo (descemos), devemo (devemos), empurramo (empurramos), encontramo (encontramos), entramo (entramos), envem (vem), envinha (vinha), escolhemo (escolhemos), esquecemo (esquecemos), estamo (estamos), estudemo (estudamos), evem (vem), falamo (falamos), fazido (feito), ficamo (ficamos), fize (fiz), fizemo (fizemos), fomo (fomos), for (formos), fraga (flagra), fragando (flagrando), frago (flagro), fumo (fomos), ganhamo (ganhamos), levamo (levamos), levantamo (levantamos), levantemo (levantamos), mandamo (mandamos), manti (mantive), o’ (olha), o’(olha), paramo (paramos), passamo (passamos), pedimo (pedimos), peguemo (pegamos), perdemo (perdemos), pinchando (pichando), pintemo (pintemos), podemo (podemos), precisamo (precisamos), pusemo (pusemos), resolvemo (resolvemos), saímo (saímos), seje (seja), sentamo (sentamos), sentemo (sentamos), separamo (separamos), somo (somos), sufro (sofro), temo (temos), tiramo (tiramos), tivemo (tivemos), tomamo (tomamos), trabalhamo (trabalhamos), trago (trazido), vesse (visse), viemo (viemos), vimo (vimos), tó (toma), cê (você), cês (vocês), e’ (ele), ea (ela), eas (elas), es (eles), ocê (você), ocês (vocês), aque’ (aquele), aquea (aquela), aqueas (aquelas), aques (aqueles), ca (com a), co (com o), cos (com os), cum (com um), cuma (com uma), d’ (de) d’(de), d’(de), d’(de), dum (de um), duma (de uma), dumas (de umas), duns (de uns),  deerreí (na DRI), ni (em), num (em um), numa (em uma), numas (em umas), pa (para), pas (para as), p’(para), p’(para), p’(para), p’(para), p’(para), p’(para), p’(para), po (para o), p’(para), pos (para os), p’(para), pra (para), pr’(para), pras (para as), pro (para o), pr’(para), pros (para os), prum (para um), pruma (para uma), pruns (para uns), p’(para), p’ (para), p’(para), pum (para um), puma (para uma), c’ aqueas (com aquelas), c’(com), c’ cê (com você), c’ e’ (com ele), c’ (com), c’(com essas), c’(com), c’ ocê (com você), c’ ocês (com vocês), daque’ (daquele), daquea (daquela), daqueas (daquelas), daques (daqueles), d’ cê (de você), de’ (dele), dea (dela), d’(de), d’(de), d’(de), des (deles), d’ es (de eles), d’(de), d’ ocê (de você), d’ ocês (de vocês), naque’ (naquele), naquea (naquela), naques (naqueles), ne’ (nele), n’ ocê (em você), n’ ocês (em vocês), p’ aque’ (para aquele), p’(para), p’ cê (para você), p’ cês (para vocês), p’ e’ (para ele), p’(para), p’ (para), p’(para), p’ es (para eles), p’ esse (para esse), p’ mim (para mim), p’ ocê (para você), p’ ocês (para vocês), pr’(para), pr’(para), pr’(para), pr’(para), pr’ ocê (para você), pr’ ocês (para vocês), p’ sio’ (para a senhora), p’ siora (para a senhora), p’(para), armoçar (almoçar), artinho (altinho), arto (alto), arto (alto), comprica (complica), compricar (complicar), cravícula (clavícula), escardada (escaldada), prano (plano), pranta (planta), pray (play), prissado (plissado), probremas (problemas), sortando (soltando), sortar (soltar), sortei (soltei), sorto (solto), sortou (soltou), vorta (volta), vortar (voltar), vortava (voltava), vorto (volto), nũ (não), canarim (canarinho), espim (espinho), padrim (padrinho), passarim (passarinho), porco-espim (porco-espinho), sozim (sozinho), almoçozim (almoçozinho), amarelim (amarelinho), azulzim (azulzinho), bebezim (bebezinho), bichim (bichinho), bocadim (bocadinho), bonitim (bonitinho), cachorrim (cachorrinho), cantim (cantinho), capoeirim (capoeirinhas), carrim (carrinho), cedezinho (CD), certim (certinho), certins (certinhos), Chapeuzim Vermelho (Chapeuzinho Vermelho), chazim (chazinho), controladim (controladinha), desfiadim (desfiadinho), direitim (direitinho), direitim (direitinho), esquisitim (esquisitinho), fechadim (fechadinho), filhotim (filhotinho), formulariozim (formulariozinho), fundim (fundinho), Geraldim (Geraldinho), golezim (golezinho), igualzim (igualzinho), instantim (instantinho), jeitim (jeitinho), Joãozim (Joãozinho), joguim (joguinho), ladim (ladinho), maciim (maciinho), mansim (mansinho), Marquim (Marquinho), meninim (menininho), morenim (moreninho), murim (murinho), Paulim (Paulinho), pequeninim (pequenininha), pertim (pertinho), negocim (negocinhos), partidim (partidinho), porquim (porquinho), portim (portinha), potim (potinho), pouquim (pouquinho), pozim (pozinho), pretim (pretinho), prontim (prontinho), quadradim (quadradinha), quadradim (quadradinho), queimadim (queimadinho), rapidim (rapidinho), recheadim (recheadinho), rolim (rolinho), tamanim (tamaninho), tampadim (tampadinho), terrenim (terreninho), tiquim (tiquinho), todim (todinho), toquim (toquinho), trancadim (trancadinhos), trenzim (trenzinho), tudim (tudinho), sio’ (senhora), sior (senhor), siora (senhora), sô (senhor), mó (muito), po’ (pode) ,tá (está) ,tamo (estamos) ,tamos (estamos) ,tão (estão) ,tar (estar) ,taria (estaria) ,tás (estás) ,tava (estava) ,tavam (estavam) ,távamos (estávamos) ,tavas (estavas) ,teja (esteja),teve (esteve) ,tive (estive) ,tiver (estiver) ,tiverem (estiverem) ,tivesse (estivesse) ,tô (estou) ,vamo (vamos) ,vão (vamos) ,vim (vir) ,xá (deixa), antiguim (antiguinho), banhozim (banhozinho), branquim (branquinho), certim (certinho), 
    devagarzim (devagarzinho), direitim (direitinho), direitim (direitinho), gostosim (gostosinho), limãozim (limãozinho), minutim (minutinho), pertim (pertinho), pulim (pulinho), pouquim (pouquinho), rapidim (rapidinho), recibim (recibinho), verdim (verdinho), xixizim (xixizinho), zerim (zerinho)
    babacar (embabacar), zucrinando (azucrinando), zucrinar (azucrinar), brigado (obrigado), brigada (obrigada), baixa (abaixa), credita (acredita), creditei (acreditei), creditou (acreditou) , baixar (abaixar), baixei (abaixei), baulado (abaulado), bora (embora), borrecido (aborrecido), brigada (obrigada), brigado (obrigado), caba (acaba), cabar (acabar), cabava (acabava), cabei (acabei), cabou (acabou), celera (acelera), celerando (acelerando), certar (acertar), chei (achei), cho (acho), contece (acontece), contecer (acontecer), conteceu (aconteceu), cordava (acordava), creditei (acreditei), dianta (adianta), doro (adoro), dotada (adotada), fessora (professora), final (afinal), fundar (afundar), garrado (agarrados), garrou (agarrou), gora (agora), gual (igual), gualzim (igualzinho), guenta (aguenta), guentando (aguentando), guentar (aguentar), guento (aguento), guentou (aguentou), inda (ainda), judar (ajudar), lambique (alambique), laranjado (alaranjado), lisou (alisou), magina (imagina), mamentar (amamentar), manhã (amanhã), marelo (amarelo), marrava (amarrava), migão (amigão aumentativo), mor (amor), ném (neném), panhava (apanhava), parece (aparece), pareceu (apareceu), partamento (apartamento), pelido (apelido),  perta (aperta), pertar (apertar), pertei (apertei), pesar (apesar), pinhada (apinhada), proveita (aproveita), proveitei (aproveitei), proveitou (aproveitou),  proveitando (aproveitando), proveitei (aproveitei), purra (empurra), qui (daqui), rancaram (arrancaram), rancava (arrancava), rancou (arrancou), ranjar (arranjar), ranjasse (arranjasse), ranjou (arranjou), rebentando (arrebentando), rebentar (arrebentar), regaço (arregaços), rorosa (horrorosa), roz (arroz), rumaram (arrumaram), sobiando (assobiando),té (até), té (até), teirinho (inteirinho), teja (esteja), tendeu (entendeu), tendi (entendi), testino (intestino), tradinha (entradinha), trapalha (atrapalha), trapalhado (atrapalhado), trapalhou (atrapalhou), travessa (atravessa), travessadinho (atravessadinho), trevida (atrevida), trevido (atrevido), trevidão (atrevidão), vó (avó), vô (avô)""" 
   

    regex_apostr = r"(\w+’(?!\n))\s(\w+(?!\n))"
    regex_excep= r"([A-Za-z]+’)([A-Za-z]+)"
    
    texto = texto.replace("'", "’")
    texto = re.sub(regex_apostr, r"\1\2", texto)
    
    formas_conv = re.sub(regex_apostr, r"\1\2", formas_conv)
    
    tuplas = re.findall(r"(\w+|\w+\s\w+|\w+’|\w+’\s?\w+|\w+\s\w+\s\w+’|\w+\s?\w+’)\s?\((\w+|\w+\s\w+)\)", formas_conv)
    tuplas=  [(x[0].strip(), x[1]) for x in tuplas]

    dicio = dict(tuplas)
        
    texto = " ".join([dicio[p] if p in dicio else p for p in texto.split(' ')])
        
    texto= re.sub(regex_excep, r'\1 \2', texto)
    
    formas_conv = re.sub(regex_excep, r"\1 \2", formas_conv)
    
    tuplas = re.findall(r"(\w+|\w+\s\w+|\w+’|\w+’\s?\w+|\w+\s\w+\s\w+’|\w+\s?\w+’)\s?\((\w+|\w+\s\w+)\)", formas_conv)
    
    dicio = dict(tuplas)
    
    texto = texto.replace('\n', '\n$ ')
    
    texto =  '\n'.join([dicio[x] if x in dicio else x for x in texto.split(' ')])
        
    texto = texto.replace('\n', ' ')
        
    texto = texto.split('$')
    texto = '\n'.join([x.strip() for x in texto])
    
    texto = texto.replace('o’', 'olha').replace('pa’', 'para')\
        .replace('Vix’', 'Vixe').replace('No’', 'Nossa').replace('pr’', 'para')\
            .replace('n’', 'não').replace('e’', 'ele').replace('Nu’', 'Nossa')
    texto = re.sub(r'i?-?_?COB_?s?r?|i?-?_?COM_?s?r?|i?-?_?APC_?s?r?|i?-?_?CMM_?s?r?|i?-?_?TOP_?s?r?=|i?-?_?TPL_?s?r?|i?-?_?APT_?s?r?|i?-?_?PAR_?s?r?=|i?-?_?PAR_?s?r?|i?-?_?INT_?s?r?|i?-?_?SCA_?s?r?|i?-?_?AUX_?s?r?|i?-?_?PHA_?s?r?|i?-?_?ALL_?s?r?|i?-?_?CNT_?s?r?|i?-?_?DCT_?s?r?|i?-?_?EXP_?s?r?|i?-?_?DCT_?s?r?', '', texto)
    texto = texto.strip()
    
    
    return texto



file_1 = '\n'.join([x for x in os.listdir() if x.endswith('xml')])
utterances = []
participant = []
start_time = []
end_time = []
   
audio = []


for filename in file_1.split():
    with open(filename, 'r', encoding="utf-8") as content:
        tree = ET.parse(content)
        root = tree.getroot()
        for y in root.iter('UNIT'):                
            utterances.append(y.text.strip())
        
            participant.append(y.get('speaker'))
            start_time.append((y.get('startTime')))
            end_time.append((y.get('endTime')))
            audio.append(filename)
            start_time_f = re.findall(r'\d+.\d+', '\n'.join(start_time))
            end_time_f = re.findall(r'\d+.\d+', '\n'.join(end_time))


df = pd.DataFrame()
df['utterances'] = utterances
df['participant'] = participant
df['audio'] = audio

df['start_time'] = start_time_f
df['end_time'] = end_time_f
df[['start_time', 'end_time']] = df[['start_time', 'end_time']].astype('float')
df['ut_length'] = round(df['end_time'] - df['start_time'], 3)
df['normalized_utterances'] = df['utterances'].apply(normaliza_coral)

with open('new_brill', 'rb') as f:
    tagueador = pickle.load(f)
    
df['normalized_utterances'] = df['normalized_utterances'].apply(lambda x: re.sub(r"o’\s?|^o’\s?|\s?o’$", 'olha', str(x), flags = re.IGNORECASE))
df['normalized_utterances'] = df['normalized_utterances'].apply(lambda x: re.sub(r"\se’\s?|^e’\s?|\s?e’$", 'ele', str(x), flags = re.IGNORECASE))
df['normalized_utterances'] = df['normalized_utterances'].apply(lambda x: re.sub(r"\sa’\s?|^a’\s?|\s?a’$", 'olha', str(x), flags = re.IGNORECASE))
df['normalized_utterances'] = df['normalized_utterances'].apply(lambda x: re.sub(r"[A-Z]{3}-r|i-[A-Z]{3}", '', str(x), flags = re.IGNORECASE))


print('Tagging with Brill tagger - Mac-Morpho')
df['utterances_POS'] = df['normalized_utterances'].apply(lambda x: nltk.word_tokenize(x)).apply(lambda x: tagueador.tag(x))

#depuração POS
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='sio’',\s')\w+|(?<='sio',\s')\w+|(?<='senhora',\s')\w+",'PROPESS', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='melancia’',\s')\w+",'N', str(x), flags = re.IGNORECASE))

df['utterances_POS'] = df['utterances_POS'].apply(lambda x:  re.sub(r"\(\'Nossa',\s\'\w+\'\),\s\(\'Senhora\',\s\'\w+\'\),", "('Nossa Senhora', 'IN'),", x))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='na',\s')\w+|(?<='no',\s')\w+", 'PREP|+', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='nessa',\s')\w+|(?<='nesse',\s')\w+|(?<='nisso',\s')\w+|(?<='nesses',\s')\w+|(?<='nessas',\s')\w+", 'PREP|+', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='ao',\s')\w+|(?<='aos',\s')\w+", 'PREP|+', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='o',\s')\w+|(?<='os',\s')\w+", 'ART', str(x), flags = re.IGNORECASE))

df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='da',\s')\w+|(?<='do',\s')\w+|(?<='dos',\s')\w+|(?<='das',\s')\w+|(?<='duma',\s')\w+|(?<='dum',\s')\w+|(?<='duns',\s')\w+|(?<='dumas',\s')\w+", 'PREP|+', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='de',\s')\w+", 'PREP', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='pelos',\s')\w+|(?<='pelo',\s')\w+|(?<='pela',\s')\w+|(?<='pelas',\s')\w+", 'PREP|+', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='dele',\s')\w+|(?<='deles',\s')\w+|(?<='dela',\s)\w+|(?<='delas',\s)\w+", 'PROADJ', str(x) , flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='num',\s')\w+", 'PREP|+', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='um',\s')\w+|(?<='uns',\s')\w+|(?<='uma',\s')\w+|(?<='umas',\s')\w+", 'ART', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='comigo',\s')\w+", 'PROPESS', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='por',\s')\w+", 'PREP', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='contigo',\s')\w+", 'PROPESS', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='ahn',\s')\w+|(?<='ham',\s')\w+|(?<='hum',\s')\w+|(?<='uhn',\s')\w+", 'IN', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='ah',\s')\w+|(?<='eh',\s')\w+|(?<='ih',\s')\w+|(?<='oh',\s')\w+|(?<='ô',\s')\w+|(?<='uai',\s')\w+|(?<='ué',\s')\w+", 'IN', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='nu',\s')\w+|(?<='pá',\s')\w+|(?<='parará',\s')\w+|(?<='tanãnãnã',\s')\w+|(?<='tchan',\s')\w+|(?<='tum',\s')\w+", 'IN', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='deixar',\s')\w+|(?<='deixa',\s')\w+|(?<='deixou',\s')\w+|(?<='deixei',\s')\w+|(?<='deixaram',\s')\w+|(?<='deixamos',\s')\w+|(?<='deixaria',\s')\w+|(?<='deixariam',\s')\w+|(?<='deixam',\s')\w+|(?<='deixo',\s')\w+|(?<='deixasse',\s')\w+|(?<='deixassem',\s')\w+|(?<='deixarmos',\s')\w+", 'VAUX', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='começar',\s')\w+|(?<='começa',\s')\w+|(?<='começou',\s')\w+|(?<='comecei',\s')\w+|(?<='começaram',\s')\w+|(?<='começam',\s')\w+|(?<='começaria',\s')\w+|(?<='começariam',\s')\w+|(?<='começamos',\s')\w+|(?<='começo',\s')\w+|(?<='começasse',\s')\w+|(?<='começassem',\s')\w+|(?<='começarmos',\s')\w+|(?<='comece',\s')\w+|(?<='comecemos',\s')\w+", 'V', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='chegar',\s')\w+|(?<='chega',\s')\w+|(?<='chegou',\s')\w+|(?<='cheguei',\s')\w+|(?<='chegaram',\s')\w+|(?<='chegam',\s')\w+|(?<='chegaria',\s')\w+|(?<='chegariam',\s')\w+|(?<='chegamos',\s')\w+|(?<='chego',\s')\w+|(?<='chegasse',\s')\w+|(?<='chegassem',\s')\w+|(?<='chegarmos',\s')\w+|(?<='chegue',\s')\w+|(?<='cheguemos',\s')\w+", 'V', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='ser',\s')\w+|(?<='é',\s')\w+|(?<='foi',\s')\w+|(?<='fui',\s')\w+|(?<='foram',\s')\w+|(?<='somos',\s')\w+|(?<='seria',\s')\w+|(?<='seriam',\s')\w+|(?<='são',\s')\w+|(?<='sou',\s')\w+|(?<='era',\s')\w+|(?<='eram',\s')\w+|(?<='for',\s')\w+|(?<='formos',\s')\w+|(?<='fosse',\s')\w+|(?<='fóssemos',\s')\w+|(?<='sermos',\s')\w+", 'VAUX', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='poder',\s')\w+|(?<='pode',\s')\w+|(?<='pôde',\s')\w+|(?<='pude',\s')\w+|(?<='puderam',\s')\w+|(?<='podemos',\s')\w+|(?<='poderia',\s')\w+|(?<='poderiam',\s')\w+|(?<='podem',\s')\w+|(?<='posso',\s')\w+|(?<='podia',\s')\w+|(?<='podiam',\s')\w+|(?<='pudesse',\s')\w+|(?<='pudessem',\s')\w+|(?<='podermos',\s')\w+", 'VAUX', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='estar',\s')\w+|(?<='está',\s')\w+|(?<='esteve',\s')\w+|(?<='estive',\s')\w+|(?<='estiveram',\s')\w+|(?<='estamos',\s')\w+|(?<='estaria',\s')\w+|(?<='estariam',\s')\w+|(?<='estão',\s')\w+|(?<='estou',\s')\w+|(?<='estava',\s')\w+|(?<='estivemos',\s')\w+|(?<='estivesse',\s')\w+|(?<='estivéssemos',\s')\w+|(?<='estivessem',\s')\w+|(?<='estarmos',\s')\w+", 'VAUX', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='ir',\s')\w+|(?<='vai',\s')\w+|(?<='foi',\s')\w+|(?<='fui',\s')\w+|(?<='foram',\s')\w+|(?<='vamos',\s')\w+|(?<='iria',\s')\w+|(?<='iriam',\s')\w+|(?<='vão',\s')\w+|(?<='vou',\s')\w+|(?<='ía',\s')\w+|(?<='fomos',\s')\w+|(?<='fosse',\s')\w+|(?<='fóssemos',\s')\w+|(?<='iria',\s')\w+|(?<='vamos',\s')\w+", 'VAUX', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='nũ',\s')\w+|(?<='né',\s')\w+|(?<='aí',\s')\w+", 'ADV', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='Whatsapp',\s')\w+|(?<='Instagram',\s')\w+|(?<='Facebook',\s')\w+|(?<='big',\s')\w+|(?<='brother',\s')\w+|(?<='buffet',\s')\w+|(?<='feedback',\s')\w+|(?<='fair',\s')\w+|(?<='play',\s')\w+", '|EST', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='open',\s')\w+|(?<='over',\s')\w+|(?<='photoshop',\s')\w+|(?<='pop',\s')\w+|(?<='plus',\s')\w+|(?<='réveillon',\s')\w+|(?<='sexy',\s')\w+|(?<='serial',\s')\w+|(?<='killer',\s')\w+", '|EST', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='shopping',\s')\w+|(?<='short',\s')\w+|(?<='show',\s')\w+|(?<='smartphone',\s')\w+|(?<='software',\s')\w+|(?<='telemarketing',\s')\w+|(?<='videogame',\s')\w+|(?<='tablet',\s')\w+|(?<='Windows',\s')\w+", '|EST', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='yes',\s')\w+|(?<='vip',\s')\w+|(?<='web',\s')\w+|(?<='smartphone',\s')\w+|(?<='slide',\s')\w+|(?<='states',\s')\w+|(?<='videogame',\s')\w+|(?<='online',\s')\w+|(?<='office',\s')\w+|(?<='offline',\s')\w+", '|EST', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='ficar',\s')\w+|(?<='fica',\s')\w+|(?<='ficou',\s')\w+|(?<='fiquei',\s')\w+|(?<='ficaram',\s')\w+|(?<='ficamos',\s')\w+|(?<='ficaria',\s')\w+|(?<='ficariam',\s')\w+|(?<='ficam',\s')\w+|(?<='fico',\s')\w+|(?<='ficasse',\s')\w+|(?<='ficassem',\s')\w+|(?<='ficarmos',\s')\w+", 'VAUX', str(x), flags = re.IGNORECASE))
df['utterances_POS'] = df['utterances_POS'].apply(lambda x: re.sub(r"(?<='né',\s')\w+", 'ADV', str(x), flags = re.IGNORECASE))


if "=PAUSA_P=" in df.utterances.values:
    try:
        
        df['TMT'] = df['utterances'].apply(lambda x: len(re.findall(r"=PAUSA_P=?", x)))
        df['corpus'] = df['audio'].apply(lambda x: "Coral_esq" if x.startswith('m') else 'Coral_brasil')
        
        df['word_before']= df['utterances_POS'].apply(lambda x: ' '.join(re.findall(r"(?<=\[)\('(\w+)", x)))
        df['class_before'] = df['utterances_POS'].apply(lambda x: ' '.join(re.findall(r"(?<=\[)\('\w+',\s'(\w+)", x)))
        
        
        pausas_index = pd.Series((df.index[df['utterances'] == '=PAUSA_P=']))
        pausas_mais_um = pd.Series((df.index[df['utterances'] == '=PAUSA_P='] + 1))
        
        valores = df['ut_length'].iloc[pausas_index]
        
        
        df['pause_len'] = valores.set_axis(pausas_mais_um)
        df['pause_len'] = df['pause_len'].apply(lambda x: 0 if pd.isnull(x) == True else x)
        
        df_pauses = df.query('pause_len > 0')
        
        df_pauses = df_pauses.query('word_before != "PAUSA_P"')
        df_pauses['word_before'] = df_pauses['word_before'].apply(lambda x: '0' if len(x) < 1 else x)
        df_pauses['class_before'] = df_pauses['class_before'].apply(lambda x: '0' if len(x) < 1 else x)
        
        df_pauses = df_pauses.query('word_before != "0"')
        df_pauses = df_pauses.query('class_before != "0"')
        
        df.to_csv('df_total.csv')
        df_pauses.to_csv('df_pausas.csv')
        
        freq_dist = pd.DataFrame(df_pauses['word_before'].value_counts())
        freq_dist.reset_index(inplace=True)
        freq_dist.columns = ['palavra_diante_pausa', 'frequência']
        freq_dist.to_csv('df_palavras_diante.csv')
        
        sns.set_style('whitegrid')
        plt.figure(dpi = 300, figsize=(7, 5))
        a = sns.histplot(data= df_pauses, x = 'pause_len', kde = True)
        a.set_title('Distribuição da duração das pausas', fontsize = 16)
        a.set_xlabel("Tempo - (s)",fontsize= 14)
        a.set_ylabel("Frequência",fontsize = 15)
        a.tick_params(labelsize=15)
        plt.show()
        
        sns.set_style('whitegrid')
        plt.figure(dpi = 200, figsize=(8, 5))
        a = sns.barplot(data = df_pauses , x = 'class_before', y = 'pause_len',  palette = 'inferno', estimator = np.mean, ci = False)
        a.set_title('Duração de pausas diante de classes de palavras', fontsize = 16)
        a.set_xlabel("Classes de palavras",fontsize= 14)
        a.set_ylabel("Média de duração",fontsize = 15)
        a.tick_params(labelsize=15)
        plt.xticks(rotation=90)
        plt.show()
        
        
        plt.figure(dpi = 200, figsize=(8, 5))
        a = sns.countplot(data = df_pauses, x = 'class_before', palette = 'inferno')
        a.set_title('Frequência de pausas diante de classes de palavras', fontsize = 16)
        a.set_xlabel("Classes de palavras",fontsize= 14)
        a.set_ylabel("Frequência",fontsize = 15)
        a.tick_params(labelsize=15)
        plt.xticks(rotation=90)
        plt.show()
        
        
        plt.figure(dpi = 200, figsize=(8, 5))
        a = sns.lineplot(data = freq_dist[:30], x = 'palavra_diante_pausa', y = 'frequência', palette = 'inferno')
        a.set_title('Palavras mais frequentes diante de pausas preenchidas', fontsize = 16)
        a.set_xlabel("Classes de palavras",fontsize= 14)
        a.set_ylabel("Frequência",fontsize = 15)
        a.tick_params(labelsize=15)
        plt.xticks(rotation=90)
        plt.show()
        
    except: 
        pass




