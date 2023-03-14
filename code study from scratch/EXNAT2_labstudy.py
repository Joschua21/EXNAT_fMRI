####################################################
#                                                  #
#   Create EXNAT-2 EEG experiment using PsychoPy   #
#                                                  #
####################################################


# Author: Merle Schuckart (Auditory Cognition Group, Dep. for Psych. 1, University of Lübeck)
# Contact: merle.schuckart@uni-luebeck.de
# Version: 1.0 (08.03.2023)



''' ---- 1. Settings ----------------------------------------'''


# Import packages 
# --> you might have to pip install some of the packages first


# os for getting/setting working directory
import os

# for getting current date & time
import datetime

import pyglet 

# psychopy for creating study components
from psychopy import visual, core, event, gui

# numpy for being able to calculate
import numpy as np

# for random number generator
import random

# for saving data in csv
import pandas as pd

# pylsl for pushing triggers to lsl stream
# from pylsl import StreamInlet, resolve_stream, StreamOutlet, StreamInfo

# I guess that's for serial ports?
# import serial 


# Set working directory:

# get the absolute path of the current file
#current_file = os.path.abspath(os.getcwd())

# get the directory containing the current file
#current_dir = os.path.dirname(current_file)

# set the working directory to the directory containing the current file
#os.chdir(current_dir)


os.chdir("/Users/merleschuckart/Github/PhD/EXNAT/EEG_study_EXNAT2/code study from scratch")


# I defined a function for generating colour lists in another script, so import that one, too.
# Also import function for counting n-back targets in stimulus list
from nback_colour_generator import create_nback_stimlist, get_targets


''' ---- 2. Main Experiment function ----------------------------------------'''

def main_experiment():
            


        ## ################    Setup LSL Stream     ####################

        # Create trigger stream:
            
        global out_marker
        #info_marker_stream = StreamInfo('PsychoPyMarkers', 'Marker', 1, 0, 'string')
        #out_marker = StreamOutlet(info_marker_stream)
        out_marker.push_sample(["TEST MARKER"])



        ## ################    Setup CSV           ####################

        # Create columns for all information we'd like to save
        
        # col 0: ID
        ID = []
        
        # col 1: age
        age = []
        
        # col 3: gender
        gender = []
        
        # col 4: handedness
        handedness = []
        
        # col 5: sender
        sender = []
        
        # col 6: block_kind
        block_kind = []
        
        # col 7: text_nr
        text_nr = []
            
        # col 8: duration
        duration = []
        
        # col 9: word
        word = []
        
        # col 10: colour
        colour = []
        
        # col 11: target
        target = []
        # col 12: nback_response
        nback_response = []
        
        # col 13: nback_RT
        nback_RT = []
        
        # col 14: trial_nr
        trial_nr = []
        
        # col 15: block_nr
        block_nr = []
        
        # col 16: Q1
        Q1 = []

        # col 17: Q2
        Q2 = []
        
        # col 18: Q3
        Q3 = []
        
        # col 19: subj_reading_effort1
        subj_reading_effort1 = []
        
        # col 20: subj_reading_effort2
        subj_reading_effort2 = []
        
        # col 21: subj_text_difficulty
        subj_text_difficulty = []
        
        # col 22: subj_text_incomprehensibility1
        subj_text_incomprehensibility1 = []
        
        # col 23: subj_text_incomprehensibility2
        subj_text_incomprehensibility2 = []
        
        # col 24: subj_interest_in_text
        subj_interest_in_text = []



        ## ################    Demographics         ####################

        # get user input
        # what do we want to know?
        exp_info = {'Geschlecht': 'w', 'Alter': '', 'Versuchspersonen-Code': '', 'Händigkeit': 'r'}
        # create dialogue box with text fields
        dlg = gui.DlgFromDict(exp_info) # show small window for user input

        # Check whether user clicked okay or cancel in the dialogue box.
        # If they clicked cancel, print message and abort experiment
        if not dlg.OK:
            print("User pressed 'Cancel'!")
            core.quit()
            
        # If everything's fine though, proceed: Push information to LSL.
        # In this new LSL stream called "Demographics", we have 7 channels, each channel containing a string (
        # = our demographical data and a few additional information on the dataset)
        demogr_info = pylsl.StreamInfo('Demographics', 'DemographicsData', 7, 0, pylsl.cf_string)
        demogr_outlet = pylsl.StreamOutlet(demogr_info)
        demogr_data = ['Participant ID: ' + exp_info['Versuchspersonen-Code'] , 
                       'Age: ' + exp_info['Alter'], 
                       'Gender: ' + exp_info['Geschlecht'], 
                       'Handedness: ' + exp_info['Handedness'], 
                       'Native Language: German',
                       'Vision: corrected or corrected to normal',
                       'Date & time of recording' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M")]
        demogr_outlet.push_sample(demogr_data)

        
        ## ################    Experiment Settings     ####################

        """ set up window for experiment """
        
        global win
        #out_marker.push_sample(["start"])
        win = visual.Window(size     = (800,600), # set window size
                            fullscr  = False, # make window full screen for better timing and less distractions
                            allowGUI = True, # False = draw window w/o frame or closing buttons
                            monitor  = 'testMonitor', 
                            units    = 'norm',
                            color    = [1,1,1]) # make the background white for a start
                
        
        
        
        """ set response keys """
        # make sure there are no key events defined so we start with a clean slate
        event.globalKeys.clear()
        
        # to go from word to word, I use the Space bar
        # to indicate that an n-back target was detected & 
        # the key C for right-handed people 
        # (we don't test left-handed people in the EEG study)

        # add new global event keys "c" and "Space"
        # "c" calls the function target_response and Space calls the 
        # function next_word (both defined at the end of the script)
        #target_key = "c"
        #event.globalKeys.add(key = target_key, func = target_response)

        #continue_key = "Space"
        #event.globalKeys.add(key = continue_key, func = next_word)


        """ set colours """        
        global colours 
        colours = ["#D292F3", "#F989A2", "#2AB7EF", "#88BA3F"]
        #  #D292F3 = weird lilac with a 2000s vibe
        #  #F989A2 = bubblegum pink
        #  #2AB7EF = Twitter blue
        #  #88BA3F = medium grass green
        # (#D8A244 = dark curry-ish yellow --> excluded!)
        
        # All colours have a luminance of 70 and a chroma of 74.
        
        # The colours are selected for distinguishability (is that a word?!) 
        # for people with "normal" colour vision as well as for 
        # people with protanomaly (red olour vision deficiency (CVD)), 
        # deuteranomaly (green CVD) and 
        # tritanomaly (blue CVD).
        
        # People with a "true" colour blindness 
        # (i.e. protanopia, deuteranopia, tritanopia)
        # shouldn't participate in this study.


        """ set texts + questions + answers """
        # Prepare lists with words & additional information on the blocks
        # hint: escape " characters like this \"

        # Giant Squid
        global text_01
        text_01 = ["Beim", "Anblick", "der", "Tiere", "wird", "klar,", "warum", "die", "Seeleute", "vergangener", "Jahrhunderte", "Angst", "vor", "Seeungeheuern", "hatten.", "Meterlange", "Fangarme,", "spitze", "Mäuler", "und", "riesige", "Augen", "verleihen", "den", "großen", "Kalmaren", "ein", "furchterregendes", "Aussehen.", "Vor", "der", "Küste", "Chiles", "sind", "sie", "aktuell", "in", "Massen", "zu", "sehen.", "Hunderte", "von", "riesigen", "Tintenfischen", "schwimmen", "in", "den", "flachen", "Gewässern", "und", "fressen", "dort", "die", "Fische.", "Normalerweise", "sind", "die", "großen", "Kalmare", "nur", "schwer", "zu", "beobachten.", "Sie", "leben", "eigentlich", "im", "offenen", "Meer.", "Nur", "nachts", "kommen", "sie", "an", "die", "Oberfläche,", "um", "kleine", "Fische", "zu", "jagen.", "Seit", "zwei", "Wochen", "aber", "sind", "sie", "auch", "bei", "Tageslicht", "vor", "der", "Küste", "zu", "sehen.", "Zunächst", "wurden", "mehr", "als", "200", "Kalmare", "vor", "einer", "Insel", "vor", "Chile", "gesichtet.", "Später", "wurden", "dann", "weitere", "Kalmare", "an", "anderen", "Orten", "entlang", "der", "Küste", "Chiles", "beobachtet.", "Vor", "allem", "für", "die", "Fischer", "ist", "das", "ärgerlich.", "Die", "Kalmare", "fressen", "Hechte,", "Sardinen", "und", "Sardellen.", "Und", "sie", "haben", "großen", "Hunger -", "schlechte", "Karten", "für", "die", "kleineren", "Fische.", "Die", "Kalmare", "selbst", "haben", "hingegen", "Glück:", "Sie", "gelten", "zwar", "in", "manchen", "Ländern", "als", "Delikatesse,", "werden", "in", "Chile", "jedoch", "nicht", "gegessen.", "Meeresbiologen", "standen", "wegen", "des", "plötzlichen", "Erscheinens", "der", "Kalmare", "zunächst", "vor", "einem", "großen", "Rätsel.", "Nun", "ist", "jedoch", "klar,", "weshalb", "die", "Kalmare", "plötzlich", "auftauchten.", "Im", "Februar", "erwärmte", "sich", "das", "Meer", "verglichen", "mit", "den", "Temperaturen", "in", "den", "Vorjahren", "ungewöhnlich", "stark.", "Es", "sammelten", "sich", "viele", "kleine", "Fische", "vor", "der", "Küste.", "Die", "Kalmare", "wurden", "dadurch", "magisch", "angezogen.", "Die", "kleinen", "Fische", "bedeuten", "für", "sie", "reiche", "Beute.", "Für", "Forscher", "ist", "das", "ein", "besonderer", "Glücksfall.", "Normalerweise", "sind", "die", "großen", "Kalmare", "so", "schwer", "vor", "die", "Kamera", "zu", "bekommen,", "dass", "Forscher", "teilweise", "auf", "seltsame", "Ideen", "kommen.", "Ein", "Biologe", "aus", "Neuseeland", "etwa", "will", "versuchen,", "den", "legendären", "Riesenkalmar", "mit", "Sexualhormonen", "vor", "die", "Linse", "zu", "locken.", "Der", "Riesenkalmar", "ist", "mit", "bis", "zu", "20", "Metern", "Länge", "und", "einer", "halben", "Tonne", "Gewicht", "das", "größte", "wirbellose", "Tier", "der", "Welt.", "Bisher", "wurde", "er", "aber", "noch", "nie", "innerhalb", "seines", "natürlichen", "Lebensraums", "gefilmt."]
        
        text_01_Q1 = "Wieso tauchten 2004 vor der Küste Chiles auf einmal hunderte Kalmare auf?"
        text_01_ans = ["sie wurden nicht mehr befischt, weshalb ihr Bestand innerhalb einer Saison quasi explodierte", "vor Chile gab es im Sommer zuvor ein großes Haisterben, weshalb sie kaum noch natürliche Fressfeinde hatten", "sie folgten kleinen Beutefischen vor die Küste", "sie wurden von Meeresbiologen vor der Küste Perus ausgewildert und migrierten dann nach Süden"]
        text_01_corr = ["a", "b", "TRUE_c", "d"]
        
        text_01_Q2 = "Wie möchte der Biologe die Kalmare zu seiner Kamera locken?"
        text_01_Q2_ans = ["mit niederfrequenten Tönen", "mit Heringen", "mit Wärmestrahlern", "mit Sexualhormonen"]
        text_01_Q2_corr = ["a", "b", "c", "TRUE_d"]
        
        text_01_Q3 = "Wie groß sind die Kalmare?"
        text_01_Q3_ans = ["20 m", "15 m", "10 m", "5 m"]
        text_01_Q3_corr = ["TRUE_a", "b", "c", "d"]
        
        
        # Hemingway / The old man and the sea
        global text_02
        text_02 = ["Zwanzig", "Jahre", "verbrachte", "Ernest", "Hemingway", "auf", "Kuba.", "Die", "meiste", "Zeit", "davon", "lebte", "er", "in", "Vigía,", "seinem", "legendären", "Haus", "vor", "den", "Toren", "Havannas.", "Seit", "seinem", "Tod", "Anfang", "der", "1960er", "lagern", "in", "Vigía", "etwa", "3000", "Manuskripte", "des", "Autors.", "In", "den", "vergangenen", "Jahren", "sind", "sie", "nach", "und", "nach", "digitalisiert", "worden.", "Nun", "will", "man", "sie", "zunächst", "in", "Kuba", "zeigen,", "bevor", "sie", "dann", "der", "Bibliothek", "von", "Boston", "übergeben", "werden.", "Die", "Leiterin", "des", "Archivs", "in", "Vigía", "sagte", "am", "Montag", "im", "kubanischen", "Fernsehen,", "es", "handele", "sich", "um", "bisher", "unveröffentlichte", "Stücke.", "Die", "Digitalisierung", "war", "zwischen", "den", "USA", "und", "Kuba", "bereits", "2002", "vereinbart", "worden.", "Damals", "berichtete", "die", "\"New", "York", "Times\",", "unter", "den", "Dokumenten", "befänden", "sich", "Texte,", "die", "auf", "lose", "Blätter", "und", "Buchrücken", "gekritzelt", "worden", "seien.", "Dies", "seien", "vor", "allem", "Briefe,", "Entwürfe", "und", "Aufzeichnungen", "zu", "Hemingways", "großen", "Romanen.", "Der", "Biograf", "Hemingways", "nannte", "den", "gesamten", "Nachlass", "eine", "\"Computertomografie", "von", "Hemingways", "Gehirn\".", "In", "Kuba", "schrieb", "Hemingway", "unter", "anderem", "seine", "berühmte", "Novelle", "\"Der", "alte", "Mann", "und", "das", "Meer\".", "Die", "Novelle", "handelt", "vom", "Kampf", "eines", "alten", "Fischers", "mit", "einem", "riesigen", "Schwertfisch.", "Zwei", "Tage", "und", "zwei", "Nächte", "ringt", "der", "Fischer", "mit", "dem", "Schwertfisch,", "bis", "er", "ihn", "schließlich", "am", "dritten", "Tag", "überwältigen", "kann.", "Da", "der", "Fisch", "zu", "groß", "ist,", "um", "ihn", "ins", "Boot", "zu", "ziehen,", "bindet", "er", "ihn", "von", "außen", "ans", "Boot.", "Das", "Blut", "des", "Schwertfisches", "lockt", "auf", "der", "Heimfahrt", "jedoch", "Haie", "an.", "Die", "Haie", "fressen", "nach", "und", "nach", "Teile", "des", "Schwertfischs,", "sodass", "dem", "Fischer", "nur", "das", "Skelett", "bleibt,", "als", "er", "wieder", "in", "den", "Heimathafen", "zurückkehrt.", "Die", "Figur", "des", "Fischers", "ist", "vermutlich", "angelehnt", "an", "den", "Fischer", "Gregorio", "Fuentes,", "der", "sich", "auf", "Kuba", "um", "Hemingways", "Boot", "kümmerte.", "Für", "sein", "Werk", "wurde", "Hemingway", "mit", "dem", "Literaturnobelpreis", "ausgezeichnet.", "Die", "Nobelpreis-Medaille", "schenkte", "er", "aus", "Verbundenheit", "zu", "Kuba", "der", "Wallfahrtskirche", "der", "barmherzigen", "Jungfrau", "von", "Cobre.", "Sie", "ist", "die", "Schutzpatronin", "Kubas.", "Die", "Medaille", "ist", "auch", "heute", "noch", "in", "der", "Kirche", "zu", "sehen."]
        
        text_02_Q1 = "Wem schenkte Hemingway seine Nobelpreis-Medaille?"
        text_02_Q1_ans = ["seiner Frau Mary Welsh Hemingway", "seiner Lieblingsbar “Sloppy Joe’s” in Key West, Florida", "einer Wallfahrtskirche zu Ehren der Schutzpatronin von Kuba", "dem Fischer Gregorio Fuentes"]
        text_02_Q1_corr = ["a", "b", "TRUE_c", "d"]
        
        text_02_Q2 = "Wie heißt Hemingways Wohnhaus auf Kuba?"
        text_02_Q2_ans = ["Vigia", "Baluarte", "Almeneas", "Fortaleza"]
        text_02_Q2_corr = ["TRUE_a", "b", "c", "d"]
        
        text_02_Q3 = "Als was bezeichnet der Hemingway-Biograf Andrew Scott Berg den Nachlass Hemingways?"
        text_02_Q3_ans = ["als ein Röntgenbild von Hemingways Seele", "als ein MRT von Hemingways Geist", "als eine Computertomografie von Hemingways Gehirn", "als ein Ultraschall von Hemingways Herz"]
        text_02_Q3_corr = ["a", "b", "TRUE_c", "d"]
        
        
        # Einstein
        global text_03
        text_03 = ["Sie", "besuchte", "ihn", "oft", "in", "seinem", "Haus", "und", "bekam", "Briefe", "und", "Gedichte", "von", "ihm.", "Manchmal", "durfte", "sie", "ihm", "sogar", "die", "Haare", "schneiden.", "Johanna", "Fantova", "galt", "als", "letzte", "Freundin", "von", "Albert", "Einstein.", "Die", "beiden", "trafen", "sich", "regelmäßig,", "telefonierten", "viel", "und", "gingen", "miteinander", "segeln.", "Nun", "wurde", "bekannt,", "was", "offenbar", "nicht", "einmal", "Einstein", "wusste:", "Johanna", "Fantova", "fertigte", "Notizen", "über", "den", "Inhalt", "ihrer", "Gespräche", "an.", "In", "ihren", "Aufzeichnungen", "zeigt", "sie", "ihn", "nicht", "als", "den", "großen", "Mann,", "der", "zu", "Lebzeiten", "zur", "Legende", "wurde,", "sondern", "als", "Einstein,", "den", "Menschenfreund.", "Die", "Nachwelt", "dürfte", "ihr", "dankbar", "sein.", "Ohne", "die", "Notizen", "wüssten", "wir", "heute", "nichts", "von", "Bibo,", "dem", "traurigen", "Papagei.", "Auch", "eine", "ganze", "Reihe", "kluger", "und", "lustiger", "Zitate", "von", "Einstein", "wären", "verloren", "gegangen.", "Es", "ist", "bisher", "das", "einzige", "bekannte", "Tagebuch", "von", "einer", "Person,", "die", "während", "seiner", "letzten", "Jahre", "eng", "mit", "Einstein", "befreundet", "war.", "Die", "22", "Jahre", "jüngere", "Johanna", "Fantova", "stammte", "wie", "Einstein", "aus", "Europa.", "Die", "Eltern", "ihres", "Ehemannes", "Otto", "Fanta", "empfingen", "vor", "dem", "Krieg", "viele", "berühmte", "Gäste", "in", "ihrem", "Salon.", "Neben", "Einstein", "zählte", "dazu", "auch", "Franz", "Kafka.", "Johanna", "Fantova", "war", "für", "Einstein", "daher", "ein", "Teil", "der", "alten", "Welt.", "Sie", "war", "eine", "Verbindung", "zu", "Dingen,", "die", "er", "vermisste.", "In", "Fantovas", "Manuskript", "erscheint", "Einstein", "als", "komischer", "Eigenbrötler.", "Zugleich", "beschreibt", "sie", "ihn", "aber", "auch", "als", "Menschenfreund,", "der", "vielen", "seiner", "Freunde", "bei", "persönlichen", "Problemen", "half.", "Und", "doch", "fühlte", "sich", "Einstein", "nie", "wirklich", "mit", "seinen", "Mitmenschen", "verbunden.", "Angesichts", "seiner", "eigenen", "gescheiterten", "Beziehungen", "betrachtete", "er", "die", "Beziehungen", "seiner", "Freunde", "mit", "spöttischer", "Distanz:", "\"Ich", "war", "bei", "einem", "Nachbarn.", "Es", "besteht", "die", "Gefahr,", "dass", "sein", "Sohn", "heiratet.\"", "Rührend", "kümmerte", "er", "sich", "dagegen", "um", "sein", "Haustier,", "einen", "deprimierten", "Papagei", "namens", "Bibo.", "\"Der", "Papagei", "ist", "noch", "ganz", "verschüchtert.", "Er", "kam", "mit", "der", "Post.\"", "Einstein", "schritt", "sofort", "zur", "Tat.", "Der", "Erfolg", "blieb", "jedoch", "leider", "aus:", "\"Der", "Papagei", "ist", "traurig.", "Ich", "versuche", "ihn", "aufzuheitern,", "aber", "er", "versteht", "leider", "meine", "Witze", "nicht.\""]
        
        text_03_Q1 = "Wie hieß der Ehemann von Einsteins Freundin Johanna?"
        text_03_Q1_ans = ["Hans Spreit", "Franz Kolar", "Kurt Lift", "Otto Fanta"]
        text_03_Q1_corr = ["a", "b", "c", "TRUE_d"]
        
        text_03_Q2 = "Welches Haustier hatte Einstein?"
        text_03_Q2_ans = ["einen verschüchterten Papagei", "einen lethargischen Kater", "eine depressive Schildkröte", "einen taubstummen Kanarienvogel"]
        text_03_Q2_corr = ["TRUE_a", "b", "c", "d"]
        
        text_03_Q3 = "Wen luden die Eltern von Johannas Ehemann neben Einstein noch in ihren Salon ein?"
        text_03_Q3_ans = ["Franz Kafka", "Robert Oppenheimer", "Theodor W. Adorno", "Werner Heisenberg"]
        text_03_Q3_corr = ["TRUE_a", "b", "c", "d"]
        
        
        # Batman's Joker
        global text_04
        text_04 = ["Jerry", "Robinson", "war", "erst", "17,", "als", "er", "die", "wichtigste", "Entscheidung", "seines", "Lebens", "traf -", "und", "möglicherweise", "seinen", "größten", "Fehler", "beging.", "Statt", "wie", "geplant", "aufs", "College", "zu", "gehen,", "ließ", "er", "sich", "von", "einem", "Mann", "namens", "Bob", "Kane", "als", "Zeichner", "engagieren.", "Kane", "hatte", "gerade", "die", "Zeichnungen", "für", "ein", "Comicheft", "abgeliefert,", "in", "dem", "er", "eine", "völlig", "neue", "Figur", "auftreten", "ließ,", "genannt", "\"Batman\".", "Jetzt", "machte", "er", "Urlaub", "in", "den", "Poconos,", "einem", "Ausflugsgebiet", "in", "Pennsylvania.", "Der", "untergewichtige", "Jerry", "Robinson", "war", "auf", "einer", "Kur", "dort,", "um", "Gewicht", "zuzulegen.", "Um", "Bob", "Kane", "von", "seinem", "Talent", "als", "Zeichner", "zu", "überzeugen,", "fertigte", "Jerry", "Robinson", "für", "ihn", "ein", "paar", "Zeichnungen", "an.", "Da", "er", "kein", "Papier", "zur", "Hand", "hatte,", "zeichnete", "er", "kurzerhand", "auf", "seiner", "Jacke.", "Beeindruckt", "stellte", "Kane", "den", "Jungen", "an.", "Bereits", "ab", "der", "dritten", "Ausgabe", "der", "Batman-Comics", "war", "er", "der", "Hauptzeichner", "der", "Serie.", "Seine", "Zeichnungen", "fertigte", "er", "vor", "allem", "nachts", "an.", "Tagsüber", "studierte", "er", "Journalistik", "in", "New", "York.", "Neben", "dem", "Zeichnen", "tat", "er", "sich", "auch", "bei", "der", "Entwicklung", "der", "Figuren", "hervor.", "Von", "ihm", "stammten", "die", "Entwürfe", "für", "Batmans", "Butler", "Alfred", "und", "Batmans", "Helfer,", "den", "jungen", "Robin.", "Fast", "zeitgleich", "hatte", "der", "Joker,", "Batmans", "Erzfeind,", "seinen", "ersten", "Auftritt", "in", "einem", "weiteren", "Heft.", "Robinson", "behauptete", "später,", "die", "Idee", "zur", "Figur", "des", "Jokers", "sei", "von", "ihm", "ausgegangen.", "Inspiriert", "wurde", "er", "dabei", "durch", "ein", "Kartenspiel,", "das", "seine", "Kollegen", "immer", "zur", "Hand", "hatten.", "Laut", "Bob", "Kane", "beruht", "der", "Entwurf", "des", "Schurken", "dagegen", "auf", "einer", "Szene", "aus", "einem", "Stummfilm.", "\"Robinson", "hatte", "nichts", "damit", "zu", "tun\",", "war", "sein", "drastisches", "Urteil.", "Hier", "rächte", "es", "sich,", "dass", "Jerry", "Robinson", "lediglich", "als", "Assistent", "engagiert", "war,", "auch", "wenn", "er", "der", "Hauptzeichner", "war.", "Kane", "dagegen", "hatte", "sich", "alle", "Rechte", "an", "den", "Figuren", "zugesichert.", "Nicht", "zuletzt", "deshalb", "begann", "Jerry", "Robinson", "ab", "1940", "nicht", "mehr", "für", "Kane,", "sondern", "für", "den", "Comic-Verlag", "direkt", "zu", "arbeiten.", "In", "dessen", "Studio", "saß", "er", "zeitweise", "neben", "\"Superman\"-Miterfinder", "Joe", "Shuster", "am", "Zeichentisch."]
        
        text_04_Q1 = "Warum war Jerry Robinson 1939 in den Poconos?"
        text_04_Q1_ans = ["er wollte einen Alkohol-Entzug machen", "er wollte sich von einer Lungenentzündung erholen", "er wollte abnehmen", "er wollte zunehmen"]
        text_04_Q1_corr = ["a", "b", "c", "TRUE_d"]
        
        text_04_Q2 = "Was studierte Jerry Robinson neben seiner Tätigkeit als Comiczeichner?"
        text_04_Q2_ans = ["Journalistik", "Grafikdesign", "Kunstgeschichte", "Modedesign"]
        text_04_Q2_corr = ["TRUE_a", "b", "c", "d"]
        
        text_04_Q3 = "Die Idee zu welcher Figur stammt angeblich von Jerrry Robinson?"
        text_04_Q3_ans = ["Catwoman", "Bane", "der Joker", "Harley Quinn"]
        text_04_Q3_corr = ["a", "b", "TRUE_c", "d"]

        
        # tiny chameleons
        global text_05
        text_05 = ["Die", "Korallenriffe", "und", "die", "sandigen", "Buchten", "sind", "perfekt", "für", "jede", "Urlaubsbroschüre.", "Insgesamt", "ist", "die", "Inselgruppe", "namens", "Nosy", "Hara", "vor", "Madagaskar", "aber", "doch", "recht", "karg.", "In", "dieser", "eher", "lebensfeindlichen", "Umgebung", "haben", "Biologen", "nun", "eine", "neue", "Tierart", "entdeckt:", "Das", "winzige", "Chamäleon", "Brookesia", "micra.", "Von", "der", "Schnauze", "bis", "zum", "Schwanzende", "misst", "es", "weniger", "als", "drei", "Zentimeter.", "Farblich", "machen", "die", "braunen", "Tierchen", "wenig", "her.", "Doch", "ihre", "winzige", "Körpergröße", "fasziniert", "die", "Forscher.", "\"Das", "ist", "nichts,", "wo", "man", "viele", "Untersuchungen", "machen", "muss.", "Man", "erkennt", "auch", "so,", "dass", "das", "etwas", "Neues", "ist\",", "sagt", "Miguel", "Vences.", "Der", "Zoologe", "berichtet", "in", "einem", "Fachartikel", "gleich", "von", "vier", "neuen", "Arten", "von", "Mini-Chamäleons.", "Laut", "den", "Forschern", "weiß", "man", "von", "den", "Tieren", "jedoch", "kaum", "mehr,", "als", "dass", "es", "sie", "gibt.", "Zu", "ihrer", "Lebensweise", "ist", "nur", "sehr", "wenig", "bekannt.", "Tagsüber", "leben", "die", "kleinen", "Chamäleons", "am", "Boden.", "Wenn", "möglich", "verbergen", "sie", "sich", "dabei", "unter", "einer", "Schicht", "Laub.", "Nachts", "geht", "es", "dann", "auf", "niedrig", "gelegene", "Äste", "zum", "Schlafen.", "Direkte", "Fressfeinde", "haben", "die", "Tierchen", "allerdings", "wohl", "eher", "nicht.", "Zu", "ihrem", "Glück,", "so", "die", "Forscher.", "Auf", "solchen", "kleinen", "Inseln", "kann", "die", "Konkurrenz", "zwischen", "den", "Tierarten", "schnell", "sehr", "groß", "werden.", "Auch", "die", "anderen", "neuen", "Chamäleon-Arten", "besiedeln", "nur", "kleine", "Gebiete", "auf", "Madagaskar.", "Durch", "die", "Zerstörung", "ihres", "Lebensraums", "sind", "sie", "jedoch", "besonders", "bedroht.", "Rund", "40", "Prozent", "der", "Reptilien-Arten", "auf", "Madagaskar", "gelten", "mittlerweile", "als", "gefährdet.", "Die", "Forscher", "befürchten,", "dass", "auch", "Brookesia", "tristis,", "eine", "weitere", "neu", "entdeckte", "Art,", "einer", "ungewissen", "Zukunft", "entgegen", "sieht.", "Zwar", "ist", "der", "Lebensraum", "des", "Chamäleons", "vor", "wenigen", "Jahren", "unter", "Schutz", "gestellt", "worden,", "doch", "die", "Abholzung", "des", "Gebiets", "hat", "seitdem", "leider", "sogar", "noch", "zugenommen.", "Mit", "der", "Wahl", "der", "Namen", "der", "neu", "entdeckten", "Chamäleon-Arten", "wollen", "sie", "auf", "die", "große", "Gefahr", "hinweisen,", "die", "den", "Chamäleons", "droht.", "Die", "Botschaft", "hinter", "den", "Namen", "Brookesia", "desperata", "und", "Brookesia", "tristis", "versteht", "man", "auch", "ohne", "Latein", "zu", "können:", "Desperata", "heißt", "verzweifelt", "und", "tristis", "so", "viel", "wie", "traurig."]
        
        text_05_Q1 = "Wo liegt die kleine Inselgruppe, auf der die Chamäleons entdeckt wurden?"
        text_05_Q1_ans = ["vor Sri Lanka", "vor Thailand", "vor Madagaskar", "vor Java"]
        text_05_Q1_corr = ["a", "b", "TRUE_c", "d"]
    
        text_05_Q2 = "Was haben die lateinischen Namen der neuentdeckten Chamäleons gemeinsam?"
        text_05_Q2_ans = ["Miguel Vences hat jede Chamäleon-Art nach je einem Forscher aus seinem Team benannt.", "Ihre Namen beinhalten alle das lateinische Wort \"minima\" für \"klein\".", "Ihre Namen beinhalten alle ein negativ konnotiertes Adjektiv wie \"desperata\" oder \"tristis\".", "Alle Chamäleon-Arten wurden nach berühmten Tierschutz-Aktivist*innen benannt."]
        text_05_Q2_corr = ["a", "b", "TRUE_c", "d"]
        
        text_05_Q3 = "Wo verstecken sich die Chamäleons tagsüber?"
        text_05_Q3_ans = ["unter Mangroven-Wurzeln", "unter Laub auf dem Boden", "unter Steinen und in Felsnischen", "unter morschen Baumstämmen"]
        text_05_Q3_corr = ["a", "TRUE_b", "c", "d"]

        
        # Mauritius
        global text_06
        text_06 = ["In", "den", "siebziger", "Jahren", "war", "Mauritius", "eine", "kleine", "Insel", "mitten", "im", "Indischen", "Ozean,", "die", "im", "Ausland", "niemand", "kannte.", "Heute", "ist", "sie", "dagegen", "weltweit", "als", "paradiesisches", "Urlaubsziel", "bekannt.", "Doch", "schon", "lange", "bevor", "die", "ersten", "Touristen", "kamen,", "war", "Mauritius", "in", "einigen", "Teilen", "der", "Welt", "sehr", "bekannt.", "Die", "Portugiesen", "waren", "die", "Ersten,", "die", "die", "Insel", "entdeckten.", "Sie", "nannten", "sie", "\"Schwaneninsel\",", "ließen", "sich", "aber", "nicht", "auf", "ihr", "nieder.", "Erst", "fast", "hundert", "Jahre", "später", "kamen", "die", "Holländer.", "Sie", "gaben", "der", "Insel", "zu", "Ehren", "des", "holländischen", "Prinzen", "Moritz", "den", "Namen", "\"Mauritius\".", "Den", "Namen", "Mauritius", "hat", "sie", "bis", "heute", "behalten.", "Die", "Holländer", "begannen,", "an", "der", "Küste", "Felder", "anzulegen.", "Sie", "brachten", "Zuckerrohr,", "Wild", "und", "Affen", "mit", "auf", "die", "Insel.", "Sie", "bauten", "Häuser", "und", "Festungen", "und", "holzten", "die", "dichten", "Wälder", "aus", "Ebenholz", "ab.", "Zu", "dieser", "Zeit", "lebten", "auf", "Mauritius", "noch", "viele", "Dodos.", "Die", "flugunfähigen", "Vögel", "hatten", "keine", "natürlichen", "Feinde", "auf", "der", "Insel", "und", "waren", "daher", "zu", "ihrem", "eigenen", "Unglück", "sehr", "zahm.", "Für", "die", "Holländer", "machte", "sie", "das", "zur", "perfekten", "Jagdbeute.", "Der", "Dodo", "wurde", "so", "stark", "bejagt,", "dass", "er", "schon", "einige", "Jahrzehnte", "nach", "Ankunft", "der", "Holländer", "vollständig", "ausgerottet", "war.", "Heute", "ist", "er", "das", "Nationalsymbol", "von", "Mauritius.", "Als", "kein", "Holz", "mehr", "zu", "holen", "und", "die", "Natur", "schwer", "geschädigt", "war,", "verließen", "die", "Holländer", "Mauritius.", "Um", "die", "Kontrolle", "der", "Insel", "brach", "ein", "erbitterter", "Krieg", "zwischen", "Briten", "und", "Franzosen", "aus.", "Aber", "wer", "lebt", "heute", "dort?", "Die", "Bevölkerung", "von", "Mauritius", "ist", "das", "Ergebnis", "einer", "Vermischung", "verschiedener", "Kulturen", "und", "Religionen.", "Viele", "der", "Menschen", "kamen", "nicht", "freiwillig.", "Die", "Holländer", "brachten", "afrikanische", "Sklaven", "mit", "auf", "die", "Insel.", "Aus", "Indien", "kamen", "Arbeiter", "für", "die", "Zuckerrohr-Plantagen", "und", "Handwerker,", "die", "beim", "Bau", "von", "Brücken", "und", "Straßen", "halfen.", "Zeitgleich", "mit", "den", "Indern", "kamen", "auch", "muslimische", "und", "chinesische", "Händler", "nach", "Mauritius. ", "Die", "Nachfahren", "von", "Indern,", "Franzosen,", "Chinesen,", "Arabern", "und", "afrikanischen", "Sklaven", "bilden", "heute", "auf", "Mauritius", "eine", "der", "wenigen", "echten", "multikulturellen", "Gesellschaften", "der", "Welt."]
        
        text_06_Q1 = "Woher stammten die Seefahrer, die den Dodo ausrotteten?"
        text_06_Q1_ans = ["Holland", "England", "Spanien", "Portugal"]
        text_06_Q1_corr = ["TRUE_a", "b", "c", "d"]
        
        text_06_Q2 = "Aus welchem Holz bestanden die Wälder der Insel?"
        text_06_Q2_ans = ["Mangoholz", "Teakholz", "Ebenholz", "Mahagoni"]
        text_06_Q2_corr = ["a", "b", "TRUE_c", "d"]
        
        text_06_Q3 = "Wie lautete der erste Name der Insel, um die es im Text geht?"
        text_06_Q3_ans = ["Schwaneninsel", "Kolibri-Insel", "Tukan-Insel", "Papageieninsel"]
        text_06_Q3_corr = ["TRUE_a", "b", "c", "d"]

        
        # Angkor Wat
        global text_07
        text_07 = ["Wer", "Angkor", "sagt,", "meint", "in", "der", "Regel", "Angkor", "Wat.", "Die", "berühmte", "Tempelanlage", "wurde", "vermutlich", "vor", "knapp", "900", "Jahren", "in", "den", "Dschungel", "Kambodschas", "gebaut.", "Seit", "mehr", "als", "hundert", "Jahren", "hat", "sich", "die", "Wissenschaft", "auf", "die", "riesigen", "Tempelanlagen", "und", "ihre", "Inschriften", "konzentriert.", "Für", "die", "Lebensweise", "der", "Bewohner", "der", "Region", "hat", "sich", "hingegen", "kaum", "jemand", "interessiert.", "Das", "Team", "um", "den", "Forscher", "Damian", "Evans", "hat", "nun", "erstmals", "eine", "Karte", "von", "Angkor", "Wat", "erstellt.", "Die", "Karte", "zeigt,", "dass", "Angkor", "Wat", "eine", "richtige", "Stadt", "war,", "nicht", "nur", "eine", "kleine", "Tempelanlage.", "Die", "Forscher", "gehen", "davon", "aus,", "dass", "Ihre", "Größe", "sogar", "New", "York", "übertroffen", "haben", "könnte.", "Damit", "ist", "\"Groß-Angkor\"", "die", "mit", "Abstand", "größte", "vorindustrielle", "Siedlung", "der", "Welt.", "Selbst", "die", "riesigen", "Städte", "der", "Maya", "erscheinen", "dagegen", "winzig.", "Die", "Forscher", "fanden", "außerdem", "heraus,", "dass", "Angkor", "eine", "hydraulische", "Stadt", "war.", "Dank", "eines", "komplizierten", "Bewässerungssystems", "konnten", "die", "mehr", "als", "eine", "Million", "Bewohner", "versorgt", "werden.", "Durch", "das", "riesige", "Netz", "aus", "Flüssen,", "Kanälen", "und", "Stauseen", "hat", "die", "mittelalterliche", "Stadt", "mehrmals", "im", "Jahr", "Reis", "ernten", "können.", "Das", "verschaffte", "den", "Bewohnern", "nicht", "nur", "volle", "Teller,", "sondern", "auch", "einen", "enormen", "Reichtum.", "Die", "Stadt", "umgab", "ein", "riesiges", "Geflecht", "aus", "Äckern,", "Häusern", "und", "Seen,", "das", "sich", "über", "mindestens", "tausend", "Quadratkilometer", "erstreckte.", "Auf", "dieser", "Fläche", "gibt", "es", "kaum", "einen", "Fleck,", "der", "nicht", "genutzt", "worden", "ist.", "Das", "Bewässerungsnetz", "war", "sogar", "dazu", "geeignet,", "den", "Reisanbau", "zu", "stärken.", "Für", "den", "Anbau", "von", "Reis", "braucht", "man", "jedoch", "extrem", "viel", "Wasser", "und", "riesige", "Flächen.", "Um", "die", "Felder", "und", "die", "künstlichen", "Seen,", "Flüsse", "und", "Kanäle", "anzulegen,", "mussten", "große", "Waldflächen", "gerodet", "werden.", "Mit", "der", "Zeit", "führte", "das", "wahrscheinlich", "zu", "riesigen", "Problemen", "wie", "Erdrutschen.", "Das", "gesamte", "System", "dürfte", "daher", "auch", "sehr", "empfindlich", "auf", "Naturkatastrophen", "reagiert", "haben.", "Insbesondere", "im", "Norden", "der", "Stadt", "fand", "man", "Spuren", "von", "hektischen", "Anpassungen", "und", "Deichbrüchen.", "Genaueres", "weiß", "man", "aber", "nicht.", "Die", "neue", "Karte", "der", "Stadt", "verrät", "aber", "zumindest,", "wo", "man", "nach", "Antworten", "suchen", "sollte."]
        
        text_07_Q1 = "Wie groß war die Stadt Angkor Wat?"
        text_07_Q1_ans = ["größer als New York", "größer als Tokyo", "größer als Delhi", "größer als São Paulo"]
        text_07_Q1_corr = ["TRUE_a", "b", "c", "d"]
        
        text_07_Q2 = "Wovon ernährten sich die Bewohner*innen Angkor Wats vornehmlich?"
        text_07_Q2_ans = ["Hirse", "Linsen", "Reis", "Mais"]
        text_07_Q2_corr = ["a", "b", "TRUE_c", "d"]
        
        text_07_Q3 = "Warum ist Angkor Wat vermutlich untergegangen?"
        text_07_Q3_ans = ["durch Massaker im Rahmen der Kolonialisierung Südostasiens", "durch eine Hungersnot", "durch die Pest", "durch Umweltzerstörung"]
        text_07_Q3_corr = ["a", "b", "c", "TRUE_d"]


        # Petra
        global text_08
        text_08 = ["Ein", "solcher", "Anblick", "lässt", "selbst", "Indiana", "Jones", "Kinnlade", "herunterklappen:", "Nach", "einem", "spektakulären", "Ritt", "durch", "eine", "teils", "nur", "zwei", "Meter", "schmale", "Schlucht", "erhebt", "sich", "vor", "dem", "Filmhelden", "eine", "riesige", "Fassade.", "Die", "Szene", "aus", "dem", "Kinofilm", "machte", "die", "antike", "Felsenstadt", "Petra", "endgültig", "weltberühmt.", "Die", "prachtvolle", "Fassade", "ist", "eine", "der", "schönsten", "Bauten", "in", "der", "Felsenstadt", "mitten", "in", "der", "Wüste", "Jordaniens.", "Die", "Metropole", "war", "einst", "die", "Hauptstadt", "der", "Nabatäer.", "Zwei", "Jahrhunderte", "beherrschten", "die", "Nabatäer", "große", "Teile", "des", "Handels", "im", "Nahen", "Osten.", "So", "reich", "und", "mächtig", "wurde", "das", "Wüstenvolk,", "dass", "es", "sogar", "die", "Römer", "herausforderte.", "Doch", "wie", "ihre", "riesige", "Hauptstadt", "mitten", "in", "der", "Wüste", "funktioniert", "hat,", "weiß", "niemand", "genau.", "Sicher", "ist,", "dass", "die", "Stadt", "in", "atemberaubend", "kurzer", "Zeit", "entstand.", "Als", "die", "Nabatäer", "ins", "heutige", "Jordanien", "vordrangen,", "war", "die", "Region", "die", "reinste", "Goldgrube.", "Bei", "Petra", "kreuzten", "sich", "mehrere", "Handelswege,", "darunter", "die", "uralte", "Weihrauchstraße.", "Binnen", "weniger", "Jahrzehnte", "entstanden", "hunderte", "Höhlen", "mit", "prunkvollen", "Fassaden", "und", "teils", "gewaltigen", "Räumen.", "Von", "der", "eigentlichen", "Stadt", "ist", "heute", "nichts", "geblieben", "außer", "Steinhaufen", "und", "Mauerreste.", "Bis", "in", "die", "siebziger", "Jahre", "glaubte", "man", "daher,", "Petra", "sei", "eine", "Stadt", "für", "die", "Toten", "und", "die", "Götter", "gewesen,", "und", "die", "Menschen", "hätten", "woanders", "gewohnt.", "Aber", "Petra", "war", "eine", "ganz", "normale", "Stadt,", "nur", "an", "einem", "unmöglichen", "Ort.", "Die", "seltenen", "Regenfälle", "nutzten", "die", "Nabatäer", "mit", "einem", "genialen", "Bewässerungssystem.", "Überall", "in", "der", "Stadt", "waren", "Wasserbecken", "in", "den", "Fels", "geschlagen.", "Viele", "Kilometer", "Wasserleitungen", "leiteten", "das", "Wasser", "zuerst", "in", "die", "Speicher", "und", "von", "dort", "aus", "zu", "den", "Bewohnern.", "Die", "vielen", "Leitungen,", "die", "den", "Regen", "einst", "von", "den", "wertvollen", "Fassaden", "fernhielten,", "wurden", "von", "den", "Nabatäern", "dauernd", "instand", "gehalten.", "Doch", "wenn", "es", "jetzt", "regnet,", "strömt", "das", "Wasser", "unkontrolliert", "die", "Fassaden", "herab.", "Für", "den", "Sandstein", "ist", "das", "eine", "Katastrophe.", "Versuche", "zur", "Rettung", "der", "Fassaden", "gab", "es,", "doch", "wirklich", "erfolgreich", "war", "bislang", "leider", "keiner.", "Die", "antike", "Felsenstadt", "wird", "daher", "wohl", "eines", "Tages", "wieder", "zu", "Sand", "zerfallen."]
        
        text_08_Q1 = "An welcher antiken Handelsroute lag die in Fels gehauene Wüstenstadt Petra?"
        text_08_Q1_ans = ["an der Safranstraße", "an der Weihrauchstraße", "an der Seidenstraße", "an der Kaschmirstraße"]
        text_08_Q1_corr = ["a", "TRUE_b", "c", "d"]
        
        text_08_Q2 = "Wie heißt das Wüstenvolk, um das es im Artikel geht?"
        text_08_Q2_ans = ["Nabatäer", "Tuareg", "Beduinen", "Garamanten"]
        text_08_Q2_corr = ["TRUE_a", "b", "c", "d"]
        
        text_08_Q3 = "Was könnte dazu führen, dass Petra schon bald völlig zerstört sein könnte?"
        text_08_Q3_ans = ["Massentourismus", "Sandstürme","Grabräuberei","Überschwemmungen"]
        text_08_Q3_corr = ["a", "b", "c", "TRUE_d"]
        
                
        # Shakespeare / Hamlet
        global text_09
        text_09 = ["William", "Shakespeare", "war", "etwa", "fünf", "Jahre", "alt,", "als", "gar", "nicht", "weit", "entfernt", "von", "seinem", "Heimatdorf", "die", "nur", "zweijährige", "Jane", "Shaxspere", "ums", "Leben", "kam.", "Das", "kleine", "Mädchen", "wollte", "Ringelblumen", "pflücken,", "die", "am", "Ufer", "eines", "Mühlteichs", "wuchsen.", "Beim", "Blumenpflücken", "rutschte", "Jane", "aus,", "fiel", "ins", "Wasser", "und", "ertrank.", "William", "Shakespeare,", "der", "etwa", "20", "Kilometer", "entfernt", "im", "Dorf", "Stratford-upon-Avon", "aufwuchs,", "sollte", "später", "zum", "größten", "Dramatiker", "aller", "Zeiten", "heranwachsen.", "Forscher", "der", "Universität", "von", "Oxford", "vermuten", "nun", "einen", "Zusammenhang", "dieses", "Unfalls", "mit", "Shakespeares", "Stück", "\"Hamlet\".", "Eine", "Nebenhandlung", "des", "Stücks", "erzählt", "die", "Geschichte", "der", "fiktiven", "Edeldame", "Ophelia,", "der", "Tochter", "eines", "Kämmerers.", "Ophelia", "wächst", "am", "dänischen", "Königshof", "auf,", "wo", "sie", "die", "Geliebte", "des", "Prinzen", "Hamlet", "wird.", "Ihre", "Beziehung", "wird", "von", "ihrem", "Vater", "und", "ihrem", "Bruder", "jedoch", "missbilligt.", "Sie", "bezweifeln,", "dass", "Hamlet", "die", "ehrliche", "Absicht", "hat,", "Ophelia", "zu", "heiraten.", "Als", "Hamlet", "aus", "Versehen", "Ophelias", "Vater", "tötet,", "verzweifelt", "sie", "und", "erleidet", "ein", "ähnliches", "Schicksal", "wie", "die", "kleine", "Jane.", "Beim", "Blumenpflücken", "an", "einem", "Bachufer", "verliert", "sie", "das", "Gleichgewicht", "und", "fällt", "in", "den", "Bach.", "Ihr", "Kleid", "saugt", "sich", "mit", "Wasser", "voll", "und", "zieht", "sie", "wie", "ein", "Gewicht", "nach", "unten.", "Ob", "ihr", "Tod", "ein", "Unfall", "ist", "oder", "sie", "sich", "mit", "Absicht", "nicht", "aus", "dem", "Wasser", "rettet,", "wird", "im", "Stück", "offengelassen.", "Die", "erstaunliche", "Verbindung", "zwischen", "realen", "Ereignissen", "und", "Shakespeares", "\"Hamlet\"", "fiel", "Historikern", "auf,", "als", "sie", "alte", "medizinische", "Akten", "untersuchten.", "Die", "Ähnlichkeit", "der", "Nachnamen", "könnte", "sogar", "darauf", "hinweisen,", "dass", "William", "und", "Jane", "Verwandte", "gewesen", "sein", "könnten.", "Feste", "Schreibweisen", "von", "Namen", "gab", "es", "zu", "Shakespeares", "Zeiten", "nicht.", "Für", "eine", "der", "Forscherinnen", "aus", "Oxford", "ist", "dieses", "Detail", "aber", "nicht", "entscheidend:", "\"Selbst", "wenn", "sie", "nicht", "verwandt", "gewesen", "sind,", "hat", "sich", "die", "Geschichte", "durch", "die", "Ähnlichkeit", "der", "Namen", "vielleicht", "in", "Shakespeares", "Kopf", "verankert.\"", "Neben", "historischen", "Grundlagen", "seien", "Shakespeares", "Stücke", "auch", "von", "Klatsch", "und", "Tratsch-Geschichten", "beeinflusst", "worden.", "Dazu", "könnte", "auch", "die", "Geschichte", "über", "den", "Tod", "von", "Jane", "Shaxspere", "gezählt", "haben."]
        
        text_09_Q1 = "Im Artikel wird beschrieben, dass ein Unfall in einem Nachbarort Shakespeare zu einem seiner bekanntesten Stücke inspiriert haben könnte. Um welches Stück handelt es sich?"
        text_09_Q1_ans = ["Othello", "King Lear", "Hamlet","Macbeth"]
        text_09_Q1_corr = ["a", "b", "TRUE_c", "d"]
        
        text_09_Q2 = "Wie hieß das Mädchen, um das es im Artikel geht?"
        text_09_Q2_ans = ["Rosalind Shaxspere", "Ann Shaxspere", "Viola Shaxspere", "Jane Shaxspere"]
        text_09_Q2_corr = ["a", "b", "c", "TRUE_d"]
        
        text_09_Q3 = "Wie alt war Shakespeare, als das Unglück passierte?"
        text_09_Q3_ans = ["ca. 5 Jahre", "ca. 10 Jahre", "ca. 15 Jahre", "ca. 20 Jahre"]
        text_09_Q3_corr = ["a", "b", "c", "d"]

    

        """ shuffle order of texts & their respective questions & answers"""
        
        # collect texts in lists
        global all_texts_list 
        all_texts_list = [text_01, text_02, text_03, text_04, text_05, text_06, text_07, text_08, text_09]

        # collect the text IDs in lists so I know which text was shown
        global all_texts_nrs_list 
        all_texts_nrs_list = ["text_01", "text_02", "text_03", "text_04", "text_05", "text_06", "text_07", "text_08", "text_09"]

        # shuffle lists:
        # IMPORTANT: set seed!
        # the seed should be random number over participants but the same within a participant
        seed = random.randint(1, 100)
        # now shuffle texts & text numbers
        random.Random(seed).shuffle(all_texts_list)
        random.Random(seed).shuffle(all_texts_nrs_list)
        

        """ Set order of blocks """
        # The first blocks should be:
        #    - reading baseline + training
        #    - click training
        #    - in random order: 1-back (2x single task training & 1x main & 1x dual task main) & 2-back (2x single task training & 1x main & 1x dual task main)
        #    - in random order: 2x reading BL main, 2x 1-back main, 2x 2-back main
        
        # this always comes first in the experiment
        Reading_BL = ["Reading_Baseline_training", "Reading_Baseline_main", "click_training"]
        
        # then you get both n-back conditions with trainings (which of them is first is randomized)
        oneback = ["1back_single_training1", "1back_single_training2", "1back_single_main", "1back_dual_main"]
        twoback = ["2back_single_training1", "2back_single_training2", "2back_single_main", "2back_dual_main"]
        
        # shuffle the order of the 2 lists
        main_blocks1 = [oneback, twoback]
        random.shuffle(main_blocks1)
        
        # flatten nested list
        main_blocks1 = [elem for sublist in main_blocks1 for elem in sublist]
        
        # now shuffle order of the last 6 main blocks:
        main_blocks2 = ["Reading_Baseline_main", "Reading_Baseline_main",  
                        "1back_dual_main", "1back_dual_main", 
                        "2back_dual_main", "2back_dual_main"]
        random.shuffle(main_blocks2)
        
        # put them all together:
        global all_blocks 
        all_blocks = Reading_BL + main_blocks1 + main_blocks2
        
        
        
        
        """ create n-back colour lists for all blocks """
                
        # The reading bl training text has 159 trials.
        
        # The click training has 6 trials. 
        
        # Then we also have 4 short training blocks à 20 trials each (5 targets)
        # 4 * single training
                
        # We have 2 single-task main blocks, 
        # one for 1-back and 1 for 2-back à 60 trials each (10 targets):
        # 2 * single main
        
        # There are 9 dual-task main blocks à 300 stimuli each (50 targets)
        # reading bl * 3
        # 1-back * 3
        # 2-back * 3
        
        # --> all in all, 15 blocks
        
        # So for every block, build a list with colour codes containing the right amount of targets.
        # The function is defined in another script bc it's super long, 
        # I import it at the beginning of this script.

        # First, create list with length of all texts. The length of the blocks is 
        # always in the same order, only the conditions change.
        blocks_textlen = [159, 300, 6, # reading bl blocks + click training
                          20, 60, 60, 300, 20, 60, 60, 300, # main blocks 1 + trainings & single tasks
                          300, 300, 300, 300, 300, 300] # main blocks 2        
        blocks_target_counts = [25, 50, 1, # reading bl blocks + click training
                                5, 10, 10, 50, 5, 10, 10, 50, # main blocks 1 + trainings & single tasks
                                50, 50, 50, 50, 50, 50]
        # Now loop this list. Check which condition we have there and the create colour list for each text.
        all_colour_lists = []
        all_target_lists = []
        for block_idx, block_length in enumerate(blocks_textlen):
            
            # get 1st letter of block name - that tells us the condition
            block_cond = all_blocks[block_idx][0]
            
            # for each condition, decide which n-back level we want to assign
            # For all no-n-back blocks, we use 1 (just for the colour list generation)
            global curr_nback_level
            if block_cond == "R":
                curr_nback_level = 1
            elif block_cond == "c":
                curr_nback_level = 1
            elif block_cond == "1":
                curr_nback_level = 1
            else: curr_nback_level = 2
                  
            # generate colour list for current block  
            global curr_colours
            curr_colours = create_nback_stimlist(nback_level = curr_nback_level, 
                                                 colour_codes = colours, 
                                                 story = ["x"] * block_length, 
                                                 target_abs_min = blocks_target_counts[block_idx], 
                                                 target_abs_max = blocks_target_counts[block_idx], 
                                                 zeroback_target = None)
        
            
            # get list of targets / non-targets
            curr_targets = get_targets(stim_list = curr_colours, 
                                       nback_level = curr_nback_level)

            # add to bigger lists
            all_colour_lists.append(curr_colours)
            all_target_lists.append(curr_targets)


        print("------ finished preparing stimuli! ------")


        ####################  START EXPERIMENT  ###################################
        

        # Show instructions for first block        
        global instr
        instr = "Im folgenden Experiment geht es darum, kurze Texte zu lesen. \nJeder Text wird Ihnen dabei Wort für Wort angezeigt. \nIm Anschluss an den Text werden Ihnen 3 Multiple-Choice-Fragen zum Inhalt des Textes gestellt.\n\n Bitte drücken Sie die Leertaste um zum nächsten Wort zu gehen. "
        displ_instr(instr)

        
        

        ####################  TRAINING BLOCK ###################################

        # explain task 1
        #txt = "Now match the two bars by pressing the force Sensor with auditive feedback"
        #presTextPsychoPy(txt)
        #event.waitKeys()




        ## ################    END EXPERIMENT    ####################

        win.close() # close window
        core.quit() # quit PsychoPy and then exit Python
        print("Ended experiment")





######################################################################




####################################################
#                                                  #
#           DEFINE EXPERIMENT COMPONENTS           #
#                                                  #
####################################################


''' function for showing instructions '''
def displ_instr(text):
    
    # create text stimulus object
    global show_instr  
    show_instr = visual.TextStim(win, text = text, font = "Times New Roman", color = "black", pos=[0,0])
    # show object in window
    show_instr.draw()                    # draw image
    win.flip()                          # show image on screen
    


    
    
''' function for displaying words '''
# show text stimulus centered on screen 
# --> ends on button press (Space)
# --> records button press of buttons "c" or "m" (depending on handedness)
# input arguments: text to display
# optional: font size & font colour

def text_trial(word, colour, target, trial_counter, nback_cond = None):
    
    """ trial settings """
    curr_nback_RT = 0
    
    
    """ create text stimulus object & show on screen """
    global text_stim  
    text_stim = visual.TextStim(win, text = word, font = "Times New Roman", color = colour, pos=[0,0])
    # show object in window
    text_stim.draw()                    # draw image
    win.flip()                          # show image on screen
    
    
    """ send word onset trigger to LSL stream """
    marker_text = "dual_main_" + nback_cond + "_" + word + "_" + colour + "_" + str(target)
    #out_marker.push_sample(["WORD_ONSET_" + marker_text])
    
    
    """ record trial onset time """
    onset_time = core.getTime()
    
    """ fixed word display: 50 ms = 3 frames """
    # wait for about 50 ms before participant is allowed to react 
    # (50 ms should be about 3 frames with a frame rate of 60 Hz aka 16.667 ms / cycle)
    num_frames = 3
    # get expected duration of next screen flip in seconds and add the number of 
    # frames you want to wait for by dividing the number of frames by the frame rate
    wait_time = win.getFutureFlipTime() + (num_frames / win.getActualFrameRate())
    # while the time we set as wait_time is not over, do nothing. After it, we can proceed.
    while core.getTime() < wait_time:
        pass  # do nothing
    
    """ record responses """

    if nback_cond == None:
        # In single-task reading blocks, participants can press the 
        # Space button to go to the next word.
        # Record response time if space was pressed.
        curr_duration = record_space()

    else:
        # In dual-task blocks & single-task n-back blocks, 
        # participants can either press the 
        # Space button to go to the next word or the C button 
        # to indicate they saw a target. 
        # Only end the component if Space was pressed, but record 
        # response times for both kinds of responses.
        curr_nback_RT, curr_duration = record_space_c
    
    
    """ send word offset trigger to LSL stream """
    # this time, also include nback RT & reading time
    marker_text = marker_text + "nback_RT:" + curr_nback_RT + "_RT:" + curr_duration 
    #out_marker.push_sample(["WORD_OFFSET_" + marker_text])
    
    
    """ get kind of n-back response (hit, miss, false alarm, correct rejection) """
    
    

    

    """ save data """

    """ increase trial counter """
    trial_counter += 1
    

    """ End current trial slide """
    word.setAutoDraw(False)
    colour.setAutoDraw(False)
    
    
    
    
    
    

    
""" Function to record button presses "C" and "Space" """
# --> for dual task
def record_space_c():
    # placeholder variables for our RTs
    curr_nback_RT = 0
    curr_duration = 0
    
    while True:
        keys = event.getKeys(keyList = ['space', 'c'], timeStamped = True)
        if keys:
            # Record the first key pressed and its response time
            key, response_time = keys[0]
            # if the key way Space, save response time in 
            # curr_duration and break the loop
            if key == 'space':
                curr_duration = response_time
                break
            # if the key was C, record response time but don't 
            # break the loop - we want to wait for Space to be pressed!
            elif key == 'c':
                curr_nback_RT = response_time
    
    # return both RTs
    return curr_nback_RT, curr_duration
       
 
""" Function to record button press "Space" """
# --> for single task   
def record_space():
    # wait for key press:    
    keys = event.getKeys(keyList = ['space'], timeStamped = True)
    # get current duration    
    key, curr_duration = keys[0]
    # return value
    return curr_duration

    
""" Function to display MC questions and record answer """

def multiple_choice(win, question, options):
    
    """
    This function presents a multiple choice question 
    and allows the the participant to use the up/down arrow keys 
    # to select an option. 
    # Once an option is selected, it will be highlighted in green. 
    # The participant can change their selection by using the arrow keys again. 
    # They can confirm their selection by pressing the return key, 
    # which will return the selected option as a string. 
    # If the participant doesn't make a selection before pressing return, 
    # nothing will happen, which means they have to choose if they want to proceed.  
    
    Parameters:
    question (str): the question text(e.g. question = "Which bird is the fastest?")
    options (list): a list of three answer options (e.g. options = ["racing pigeon","peregrine falcon", "great philippine eagle"])
    win (visual.window.Window): window to present the question in
    
    """
    
    # Create a list of text objects for each answer option
    option_texts = [visual.TextStim(win, text = opt,  height=20, pos = (0, -50*(i-1))) for i, opt in enumerate(options, start = 1)]
    # Create a text object for the question
    question_text = visual.TextStim(win, text = question, color = "black", pos = (0, 100), height = 30)
    
    # Set initial option selection to None
    selected_option = "NO_ANS"
    
    # Create a loop to handle input
    while True:
        
        # Draw the question and options to the screen
        question_text.draw()
        
        for text_obj in option_texts:
            # Highlight the selected option in green
            if text_obj.text == selected_option:
                text_obj.color = "green"
            else:
                text_obj.color = "black"
            text_obj.draw()
        win.flip()
        
        # Wait for user input
        keys = event.waitKeys(keyList=["up", "down", "return"])
        print(keys)
        if keys == ['up']:
            # Select the previous option
            if selected_option != "NO_ANS":
                index = options.index(selected_option)
                if index > 0:
                    selected_option = options[index-1]
        
        elif keys == ['down']:
            print("detected down")
            # Select the next option
            if selected_option != "NO_ANS":
                index = options.index(selected_option)
                if index < len(options)-1:
                    selected_option = options[index+1]
        
        # force participant to answer
        elif keys == ['return'] and selected_option != "NO_ANS":
            print("detected return")
            # End the loop and return the selected option
            return selected_option
    
    
# Set up the window
win = visual.Window(size=(800, 600), color="#ffffff", monitor = "testmonitor", units="pix", fullscr=False)

# Call the function with your desired question and options
question = "Which bird is the fastest?"
options = ["racing pigeon","peregrine falcon", "great philippine eagle"]
selected_option = multiple_choice(win, question, options)

# Close the window when you're done
win.close()
core.quit()
print("closed window")



import sys
print(sys.executable)



    