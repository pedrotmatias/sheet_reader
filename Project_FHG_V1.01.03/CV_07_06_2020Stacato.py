import numpy as np 
import cv2 as cv2
import matplotlib.pyplot as plt
import scipy.signal as sig
import pygame.midi
import time
import mingus.core.notes as notes
from mingus.containers import Note


def remove_lines(sheet):
    kernel = np.ones((3, 1), np.uint8)
    notes = cv2.morphologyEx(sheet, cv2.MORPH_OPEN, kernel)
    return notes


def determine_lines(image):
    height, width = image.shape[:2]

    histV=[]
    for l in range(height):
        conta=0
        for c in range(width):
            conta+=image[l, c]
        histV.append(conta)

    HV = np.zeros((height,101), np.uint8) # altura, largura
    for l in range(height):
        cv2.line(HV, (0, l) ,(int(histV[l]*100/(255 * width)), l), 255) #(direita, baixo)


    peaks = sig.find_peaks(histV, (255 * width / 3) ) #Função para calcular o peaks, Necessário installar a SciPy ( ver na net o comand), depois fazer o Import como na linha 4

    return peaks

def remove_excess_lines(line_location):

    #takes the mean spacing between all lines
    acc = 0
    for index in range(len(line_location) - 1):
        acc = acc + (line_location[index + 1] - line_location[index])
    mean_dist = acc / (len(line_location) - 1)
    #-----------------------------------------

    #removes any line that isn't part of the music sheet
    line_counter = 4
    new_lines = list()
    for index in range(len(line_location) - 1):
        #if one line is spaced higher than the threshold to the next line, remove it
        if (line_location[index + 1] - line_location[index]) < mean_dist:
            new_lines.append(line_location[index])
            line_counter = line_counter - 1
        
        #special case for the last line of every group
        if not line_counter:
            new_lines.append(line_location[index + 1])
            line_counter = 4
    #--------------------------------------------------

    #takes the new line spacing mean, now only with the music sheet lines
    acc = 0
    mean_dist = 0
    for index in range(4):
        acc = acc + (new_lines[index + 1] - new_lines[index])
    mean_dist = acc / 4
    #-------------------------------------------------------------------

    return new_lines, int(mean_dist)



def takeSecond(elem):
    return elem[0]


def templateMatchClefsInbound(img_gray, clef_template, threshold, Staff_Size):
    _ , T_height = clef_template.shape[::-1]                #The "::-1" sintax means that it returns the values Height and Width in reverse order -> Width, height
    scale_percentage = ( Staff_Size / T_height )
    scaled_width = int( clef_template.shape[1] * scale_percentage )
    scaled_height = int( clef_template.shape[0] * scale_percentage )
    dim = ( scaled_width, scaled_height )
    resized = cv2.resize(clef_template, dim, interpolation = cv2.INTER_AREA)
    clef_template = resized

    res = cv2.matchTemplate(img_gray, clef_template, cv2.TM_CCOEFF_NORMED)
    location = np.where(res >= threshold)
    return location


def templateMatchClefsOutbound(img_gray, clef_template, threshold, Staff_Size, line_template):
    _ , T_height = line_template.shape[::-1]                #The "::-1" sintax means that it returns the values Height and Width in reverse order -> Width, height
    scale_percentage = ( Staff_Size / T_height )
    scaled_width = int( clef_template.shape[1] * scale_percentage )
    scaled_height = int( clef_template.shape[0] * scale_percentage )
    dim = ( scaled_width, scaled_height )
    resized = cv2.resize(clef_template, dim, interpolation = cv2.INTER_AREA)
    clef_template = resized

    res = cv2.matchTemplate(img_gray, clef_template, cv2.TM_CCOEFF_NORMED)
    location = np.where(res >= threshold)
    return location


def templateMatchNotes(img_gray, note_template, threshold, Staff_Size, line_template):
    _ , T_height = line_template.shape[::-1]                #The "::-1" sintax means that it returns the values Height and Width in reverse order -> Width, height
    scale_percentage = ( Staff_Size / T_height )
    scaled_width = int( note_template.shape[1] * scale_percentage )
    scaled_height = int( note_template.shape[0] * scale_percentage )
    dim = ( scaled_width, scaled_height )
    resized = cv2.resize(note_template, dim, interpolation = cv2.INTER_AREA)
    note_template = resized
    res = cv2.matchTemplate(img_gray, note_template, cv2.TM_CCOEFF_NORMED)
    location = np.where(res >= threshold)
    cv2.imshow('img', note_template)
    return location



#---------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------CODE BEGIN-------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------
font = cv2.FONT_HERSHEY_COMPLEX
fontScale = 0.5
color = (255, 0, 255)
thickness = 1


#raw_img = cv2.imread('../Data/Images/bday.jpg')
#copy_raw_img = cv2.imread('../Data/Images/bday.jpg')


raw_img = cv2.imread('../Data/Images/WANO.png')
copy_raw_img = cv2.imread('../Data/Images/WANO.png')


# raw_img = cv2.imread('../Data/Images/FHG.png')
# copy_raw_img = cv2.imread('../Data/Images/FHG.png')

cv2.imshow('s', raw_img)
grey_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
bin_inv_img = cv2.bitwise_not(grey_img)
workable_img = cv2.adaptiveThreshold(bin_inv_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -1)
Alto_clef_template = cv2.imread('../Data/Notation_image_library_small/Clefs/Alto_Clef.png', 0 )
Treble_clef_template = cv2.imread('../Data/Notation_image_library_small/Clefs/Treble_Clef.png', 0 )
Tenor_clef_template = cv2.imread('../Data/Notation_image_library_small/Clefs/Tenor_Clef.png', 0 )
Bass_clef_template = cv2.imread('../Data/Notation_image_library_small/Clefs/Bass_Clef.png', 0 )
Tenor_Lines = cv2.imread('../Data/Notation_image_library_small/Clefs/Tenor_Lines.png', 0 )
Treble_Lines = cv2.imread('../Data/Notation_image_library_small/Clefs/Treble_Lines.png', 0 )



minims_up_template = cv2.imread( '../Data/Templates/Minims/Minim.png', 0 )
minims_dn_template = cv2.imread( '../Data/Templates/Minims/Minim stem down.png', 0 )

Quaver_up_start_template = cv2.imread( '../Data/Templates/Quavers/Quavers up start.png', 0 )
Quaver_dn_start_template = cv2.imread( '../Data/Templates/Quavers/Quavers down start.png', 0 )
Quaver_up_mid_template = cv2.imread( '../Data/Templates/Quavers/Quavers up mid.png', 0 )
Quaver_dn_mid_template = cv2.imread( '../Data/Templates/Quavers/Quavers down mid.png', 0 )
Quaver_up_end_template = cv2.imread( '../Data/Templates/Quavers/Quavers up end.png', 0 )
Quaver_dn_end_template = cv2.imread( '../Data/Templates/Quavers/Quavers down end.png', 0 )
Quaver_slope_start_template = cv2.imread( '../Data/Templates/Quavers/Quavers slope start.png', 0 )
Quaver_slope_end_template = cv2.imread( '../Data/Templates/Quavers/Quavers slope end.png', 0 )

Lineless_quaver_up_start_template = cv2.imread( '../Data/Templates/Quavers/Headless/Quavers up start.png', 0 )
Lineless_quaver_dn_start_template = cv2.imread( '../Data/Templates/Quavers/Headless/Quavers down start.png', 0 )
Lineless_quaver_up_mid_template = cv2.imread( '../Data/Templates/Quavers/Headless/Quavers up mid.png', 0 )
Lineless_quaver_dn_mid_template = cv2.imread( '../Data/Templates/Quavers/Headless/Quavers down mid.png', 0 )
Lineless_quaver_up_end_template = cv2.imread( '../Data/Templates/Quavers/Headless/Quavers up end.png', 0 )
Lineless_quaver_dn_end_template = cv2.imread( '../Data/Templates/Quavers/Headless/Quavers down end.png', 0 )

Quaver_up_single = cv2.imread( '../Data/Templates/Quavers/Quaver single.png', 0 )
Quaver_dn_single = cv2.imread( '../Data/Templates/Quavers/Quaver single down.png', 0 )

S_quaver_up_start_template = cv2.imread( '../Data/Templates/Semiquavers/Semiquavers up start.png', 0 )
S_quaver_dn_start_template = cv2.imread( '../Data/Templates/Semiquavers/Semiquavers down start.png', 0 )
S_quaver_up_mid_template = cv2.imread( '../Data/Templates/Semiquavers/Semiquavers up mid.png', 0 )
S_quaver_dn_mid_template = cv2.imread( '../Data/Templates/Semiquavers/Semiquavers down mid.png', 0 )
S_quaver_up_end_template = cv2.imread( '../Data/Templates/Semiquavers/Semiquavers up end.png', 0 )
S_quaver_dn_end_template = cv2.imread( '../Data/Templates/Semiquavers/Semiquavers down end.png', 0 )

Linelees_s_quaver_up_start_template = cv2.imread( '../Data/Templates/Semiquavers/Headless/Semiquavers up start.png', 0 )
Linelees_s_quaver_dn_start_template = cv2.imread( '../Data/Templates/Semiquavers/Headless/Semiquavers down start.png', 0 )
Linelees_s_quaver_up_mid_template = cv2.imread( '../Data/Templates/Semiquavers/Headless/Semiquavers up mid.png', 0 )
Linelees_s_quaver_dn_mid_template = cv2.imread( '../Data/Templates/Semiquavers/Headless/Semiquavers down mid.png', 0 )
Linelees_s_quaver_up_end_template = cv2.imread( '../Data/Templates/Semiquavers/Headless/Semiquavers up end.png ', 0 )
Linelees_s_quaver_dn_end_template = cv2.imread( '../Data/Templates/Semiquavers/Headless/Semiquavers down end.png', 0 )


crotchets_up_template = cv2.imread('../Data/Templates/Crotchets/Crotchet.png', 0)
crotchets_dn_template = cv2.imread('../Data/Templates/Crotchets/Crotchet stem down.png', 0)

Rest_crotchets_template = cv2.imread( '../Data/Templates/Rests/Rest crotchet.png', 0 )
Rest_quaver_template = cv2.imread( '../Data/Templates/Rests/Rest quaver.png', 0 )
Accidental_sharp = cv2.imread('../Data/Templates/Accidentals/Sharp.png', 0)
Accidental_natural = cv2.imread('../Data/Templates/Accidentals/Natural.png', 0)
Accidental_flat = cv2.imread('../Data/Templates/Accidentals/Flat.png', 0)
Dot_template = cv2.imread('../Data/Templates/Dot.png', 0)

#--------------------Determine line location---------------------
blank = np.zeros_like(raw_img)
height, width = raw_img.shape[:2]
line_location = determine_lines(workable_img)

#remove lines that aren't part of the music sheet
sheet_lines, line_spacing = remove_excess_lines(line_location[0])


for peak in sheet_lines:
    cv2.line(raw_img,(0, peak), (width, peak), (0, 255, 0) , 1)
for peak in sheet_lines:
    cv2.line(blank,(0, peak), (width, peak), (0, 255, 0) , 1)


#-----------------------------------------------------------------




#-----------------Interpret notes--------------------

#filters lines out of the music sheet
lineless_sheet = remove_lines(workable_img)

#find circles that make up the notes
note_kernel = cv2.imread('../Data/Images/kernel.png')
note_kernel = cv2.cvtColor(note_kernel, cv2.COLOR_BGR2GRAY)
__, note_kernel = cv2.threshold(note_kernel, 127, 255, cv2.THRESH_BINARY_INV)
note_kernel = np.array(note_kernel)
note_sheet = cv2.morphologyEx(lineless_sheet, cv2.MORPH_OPEN, note_kernel)


circles = cv2.HoughCircles(note_sheet, cv2.HOUGH_GRADIENT, 1, int(line_spacing), param1= 30, param2= 7, minRadius= 3, maxRadius= 15)
#for i in circles[0]:
    #cv2.circle(raw_img, (i[0], i[1]), 2, (255, 255, 255), -1)


#divide sheet note by note
nr_of_staffs = int(len(sheet_lines) / 5)
snip_list = []
note_list = []
accidental_list = []
snip_boundry = 3
Staff_Size2 = sheet_lines[4] - sheet_lines[0]

for i in range(nr_of_staffs):
    root_line_top = sheet_lines[i*5]
    root_line_bottom = sheet_lines[i*5 + 4]
    middle_line = sheet_lines[i*5 + 2]
    h_min = sheet_lines[i*5] - line_spacing * snip_boundry
    h_max = sheet_lines[4 + i*5] + line_spacing * snip_boundry
    stave_height = sheet_lines[4 + i*5] - sheet_lines[i*5]
    head_width = int(42 * stave_height / 123 + 0.5) #big brain cast

    inv_lineless_sheet = cv2.bitwise_not(lineless_sheet)
    staff_snippet = grey_img[h_min:h_max, :]

    #finds notes on the current staff---------------------
    notes_in_staff = list()
    try:
        for i in circles[0]:
            if i[1] < h_max and i[1] > h_min:
                notes_in_staff.append(i)

        notes_in_staff.sort(key=takeSecond)

    except:
        print(">> Didn't find any circles")
    #--------------------------

    #Determines what line each note is on-----------------
    for i in notes_in_staff:
        if i[1] > middle_line:
            position = int(round((root_line_bottom - i[1]) / (line_spacing / 2)))
            note_list.append(position - 4)
            accidental_list.append('N')
        
        else:
            position = int(round((root_line_top - i[1]) / (line_spacing / 2)))
            note_list.append(position + 4)
            accidental_list.append('N')
    #--------------------------


    #cuts staffs into individual notes--------------------
    cut = 0
    last_cut = 0
    try:
        last_note_x = notes_in_staff[0][0]
        for i in range(len(notes_in_staff)):
            #cuts to just before the note
            cut = int(notes_in_staff[i][0] - int(head_width * 0.67 + 0.5))
            snip_list.append(staff_snippet[:, last_cut:cut])
            cv2.rectangle(raw_img, (last_cut, h_min), (cut, h_max), (0, 0, 255), 3)
            last_cut = cut

            #cuts right before and after the note
            cut = int(notes_in_staff[i][0] + int(head_width * 0.75 + 0.5))
            snip_list.append(staff_snippet[:, last_cut:cut])
            cv2.rectangle(raw_img, (last_cut, h_min), (cut, h_max), (0, 255, 0), 3)
            last_cut = cut


            last_note_x = notes_in_staff[i][0]
        
        #snip_list.append(staff_snippet[:, int(last_cut):])
        #cv2.rectangle(raw_img, (last_cut, h_min), (width, h_max), (0, 0, 255), 3)
        #raw_img = cv2.putText(raw_img, str(note_list[len(note_list)-1]), ( int( notes_in_staff[len(notes_in_staff)-1][0] ), int( h_min - (2*line_spacing)) ), font, fontScale, color, thickness, cv2.LINE_AA)
    
    except:
        print(">> same thing")
        #--------------------------



scaling = Staff_Size2 / 123
timing_list = []
minin_time = 2
crotchet_time = 1
quaver_time = 1/2
semiquaver_time = 1/4
St = 0

counter = 0
for snippet in snip_list:
    counter += 1
    St = 0
    if ( snippet.shape[1] < ( scaling * 55 ) ):
        continue
    timing_cell = []
    if( counter % 2):
        number_of_quavers = templateMatchNotes(snippet, Rest_quaver_template , 0.65, Staff_Size2, Treble_Lines)
        width = int( Quaver_up_start_template.shape[1] * scaling )
        height = int ( Quaver_up_start_template.shape[0] * scaling)
        if( len( number_of_quavers[0] ) != 0 ):
            x_pos, y_pos = number_of_quavers[1][0], number_of_quavers[0][0]
            cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
            snippet = cv2.putText(snippet, 'P', (x_pos , y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
            timing_cell.append("P")
            timing_cell.append(quaver_time)
            timing_list.append(timing_cell)

        number_of_quavers = templateMatchNotes(snippet, Accidental_sharp , 0.7, Staff_Size2, Treble_Lines)
        width = int( Quaver_up_start_template.shape[1] * scaling )
        height = int ( Quaver_up_start_template.shape[0] * scaling)
        if( len( number_of_quavers[0] ) != 0 ):
            x_pos, y_pos = number_of_quavers[1][0], number_of_quavers[0][0]
            cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
            snippet = cv2.putText(snippet, 'Sh', (x_pos , y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
            int_cont = int( (counter - 2) / 2 )
            while( int_cont < len(accidental_list) -1 ):
                if( note_list[ int_cont + 1] == note_list[ int( (counter - 1) / 2 ) ] ):
                    accidental_list[ int_cont + 1 ] = 'S'
                int_cont += 1
            continue

        number_of_quavers = templateMatchNotes(snippet, Accidental_natural , 0.7, Staff_Size2, Treble_Lines)
        width = int( Quaver_up_start_template.shape[1] * scaling )
        height = int ( Quaver_up_start_template.shape[0] * scaling)
        if( len( number_of_quavers[0] ) != 0 ):
            x_pos, y_pos = number_of_quavers[1][0], number_of_quavers[0][0]
            cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
            snippet = cv2.putText(snippet, 'N', (x_pos , y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
            int_cont = int( (counter - 2) / 2 )
            while( int_cont < len(accidental_list) -1 ):
                if( note_list[ int_cont + 1] == note_list[ int( (counter - 1) / 2 ) ] ):
                    accidental_list[ int_cont + 1 ] = 'N'
                int_cont += 1
            continue

        number_of_quavers = templateMatchNotes(snippet, Accidental_flat , 0.7, Staff_Size2, Treble_Lines)
        width = int( Quaver_up_start_template.shape[1] * scaling )
        height = int ( Quaver_up_start_template.shape[0] * scaling)
        if( len( number_of_quavers[0] ) != 0 ):
            x_pos, y_pos = number_of_quavers[1][0], number_of_quavers[0][0]
            cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
            snippet = cv2.putText(snippet, 'F', (x_pos , y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
            int_cont = int( (counter - 2) / 2 )
            while( int_cont < len(accidental_list) -1 ):
                if( note_list[ int_cont + 1] == note_list[ int( (counter - 1) / 2 ) ] ):
                    accidental_list[ int_cont + 1 ] = 'F'
                int_cont += 1
            continue

        continue

    
    number_of_quavers = templateMatchNotes(snippet, Dot_template , 0.8, Staff_Size2, Treble_Lines)
    width = int( Dot_template.shape[1] * scaling )
    height = int ( Dot_template.shape[0] * scaling)
    if( len( number_of_quavers[0] ) != 0 ):
        x_pos, y_pos = number_of_quavers[1][0], number_of_quavers[0][0]
        cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
        St = 1

    #Semiquavers-------------------------------------------------------------------------------------------------------------------------------
    number_of_quavers = templateMatchNotes(snippet, S_quaver_up_start_template , 0.7, Staff_Size2, Treble_Lines)
    width = int( Quaver_up_start_template.shape[1] * scaling )
    height = int ( Quaver_up_start_template.shape[0] * scaling)
    if( len( number_of_quavers[0] ) != 0 ):
        x_pos, y_pos = number_of_quavers[1][0], number_of_quavers[0][0]
        cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
        snippet = cv2.putText(snippet, 'S', (x_pos , y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
        if( St == 1 ):
            timing_cell.append("ST")
        else:
            timing_cell.append("N")
        timing_cell.append(semiquaver_time)
        timing_list.append(timing_cell)
        continue

    number_of_quavers = templateMatchNotes(snippet, S_quaver_dn_start_template , 0.7, Staff_Size2, Treble_Lines)
    width = int( Quaver_dn_start_template.shape[1] * scaling )
    height = int ( Quaver_dn_start_template.shape[0] * scaling)
    if( len( number_of_quavers[0] ) != 0 ):
        x_pos, y_pos = number_of_quavers[1][0], number_of_quavers[0][0]
        cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
        snippet = cv2.putText(snippet, 'S', (x_pos, y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
        if( St == 1 ):
            timing_cell.append("ST")
        else:
            timing_cell.append("N")
        timing_cell.append(semiquaver_time)
        timing_list.append(timing_cell)
        continue

    number_of_quavers = templateMatchNotes(snippet, S_quaver_up_mid_template , 0.7, Staff_Size2, Treble_Lines)
    width = int( Quaver_up_mid_template.shape[1] * scaling )
    height = int ( Quaver_up_mid_template.shape[0] * scaling)
    if( len( number_of_quavers[0] ) != 0 ):
        x_pos, y_pos = number_of_quavers[1][0], number_of_quavers[0][0]
        cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
        snippet = cv2.putText(snippet, 'S', (x_pos, y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
        timing_cell.append("N")
        timing_cell.append(semiquaver_time)
        timing_list.append(timing_cell)
        continue

    number_of_quavers = templateMatchNotes(snippet, S_quaver_dn_mid_template , 0.7, Staff_Size2, Treble_Lines)
    width = int( Quaver_dn_mid_template.shape[1] * scaling )
    height = int ( Quaver_dn_mid_template.shape[0] * scaling)
    if( len( number_of_quavers[0] ) != 0 ):
        x_pos, y_pos = number_of_quavers[1][0], number_of_quavers[0][0]
        cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
        snippet = cv2.putText(snippet, 'S', (x_pos, y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
        if( St == 1 ):
            timing_cell.append("ST")
        else:
            timing_cell.append("N")
        timing_cell.append(semiquaver_time)
        timing_list.append(timing_cell)
        continue

    number_of_quavers = templateMatchNotes(snippet, S_quaver_up_end_template , 0.7, Staff_Size2, Treble_Lines)
    width = int( Quaver_up_end_template.shape[1] * scaling )
    height = int ( Quaver_up_end_template.shape[0] * scaling)
    if( len( number_of_quavers[0] ) != 0 ):
        x_pos, y_pos = number_of_quavers[1][0], number_of_quavers[0][0]
        cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
        snippet = cv2.putText(snippet, 'S', (x_pos, y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
        if( St == 1 ):
            timing_cell.append("ST")
        else:
            timing_cell.append("N")
        timing_cell.append(semiquaver_time)
        timing_list.append(timing_cell)
        continue

    number_of_quavers = templateMatchNotes(snippet, S_quaver_dn_end_template , 0.7, Staff_Size2, Treble_Lines)
    width = int( Quaver_dn_end_template.shape[1] * scaling )
    height = int ( Quaver_dn_end_template.shape[0] * scaling)
    if( len( number_of_quavers[0] ) != 0 ):
        x_pos, y_pos = number_of_quavers[1][0], number_of_quavers[0][0]
        cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
        snippet = cv2.putText(snippet, 'S', (x_pos, y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
        if( St == 1 ):
            timing_cell.append("ST")
        else:
            timing_cell.append("N")
        timing_cell.append(semiquaver_time)
        timing_list.append(timing_cell)
        continue

    number_of_quavers = templateMatchNotes(snippet, Linelees_s_quaver_up_start_template , 0.75, Staff_Size2, Treble_Lines)
    width = int( Quaver_up_start_template.shape[1] * scaling )
    height = int ( Quaver_up_start_template.shape[0] * scaling)
    if( len( number_of_quavers[0] ) != 0 ):
        x_pos, y_pos = number_of_quavers[1][0], number_of_quavers[0][0]
        cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
        snippet = cv2.putText(snippet, 'S', (x_pos , y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
        if( St == 1 ):
            timing_cell.append("ST")
        else:
            timing_cell.append("N")
        timing_cell.append(semiquaver_time)
        timing_list.append(timing_cell)
        continue

    number_of_quavers = templateMatchNotes(snippet, Linelees_s_quaver_dn_start_template , 0.75, Staff_Size2, Treble_Lines)
    width = int( Quaver_dn_start_template.shape[1] * scaling )
    height = int ( Quaver_dn_start_template.shape[0] * scaling)
    if( len( number_of_quavers[0] ) != 0 ):
        x_pos, y_pos = number_of_quavers[1][0], number_of_quavers[0][0]
        cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
        snippet = cv2.putText(snippet, 'S', (x_pos, y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
        if( St == 1 ):
            timing_cell.append("ST")
        else:
            timing_cell.append("N")
        timing_cell.append(semiquaver_time)
        timing_list.append(timing_cell)
        continue

    number_of_quavers = templateMatchNotes(snippet, Linelees_s_quaver_up_mid_template , 0.75, Staff_Size2, Treble_Lines)
    width = int( Quaver_up_mid_template.shape[1] * scaling )
    height = int ( Quaver_up_mid_template.shape[0] * scaling)
    if( len( number_of_quavers[0] ) != 0 ):
        x_pos, y_pos = number_of_quavers[1][0], number_of_quavers[0][0]
        cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
        snippet = cv2.putText(snippet, 'S', (x_pos, y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
        if( St == 1 ):
            timing_cell.append("ST")
        else:
            timing_cell.append("N")
        timing_cell.append(semiquaver_time)
        timing_list.append(timing_cell)
        continue

    number_of_quavers = templateMatchNotes(snippet, Linelees_s_quaver_dn_mid_template , 0.75, Staff_Size2, Treble_Lines)
    width = int( Quaver_dn_mid_template.shape[1] * scaling )
    height = int ( Quaver_dn_mid_template.shape[0] * scaling)
    if( len( number_of_quavers[0] ) != 0 ):
        x_pos, y_pos = number_of_quavers[1][0], number_of_quavers[0][0]
        cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
        snippet = cv2.putText(snippet, 'S', (x_pos, y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
        if( St == 1 ):
            timing_cell.append("ST")
        else:
            timing_cell.append("N")
        timing_cell.append(semiquaver_time)
        timing_list.append(timing_cell)
        continue
    
    
    number_of_quavers = templateMatchNotes(snippet, Linelees_s_quaver_up_end_template , 0.75, Staff_Size2, Treble_Lines)
    width = int( Quaver_up_end_template.shape[1] * scaling )
    height = int ( Quaver_up_end_template.shape[0] * scaling)
    if( len( number_of_quavers[0] ) != 0 ):
        x_pos, y_pos = number_of_quavers[1][0], number_of_quavers[0][0]
        cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
        snippet = cv2.putText(snippet, 'S', (x_pos, y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
        if( St == 1 ):
            timing_cell.append("ST")
        else:
            timing_cell.append("N")
        timing_cell.append(semiquaver_time)
        timing_list.append(timing_cell)
        continue

    number_of_quavers = templateMatchNotes(snippet, Linelees_s_quaver_dn_end_template , 0.75, Staff_Size2, Treble_Lines)
    width = int( Quaver_dn_end_template.shape[1] * scaling )
    height = int ( Quaver_dn_end_template.shape[0] * scaling)
    if( len( number_of_quavers[0] ) != 0 ):
        x_pos, y_pos = number_of_quavers[1][0], number_of_quavers[0][0]
        cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
        snippet = cv2.putText(snippet, 'S', (x_pos, y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
        if( St == 1 ):
            timing_cell.append("ST")
        else:
            timing_cell.append("N")
        timing_cell.append(semiquaver_time)
        timing_list.append(timing_cell)
        continue


    #Quavers-------------------------------------------------------------------------------------------------------------------------------------------
    
    number_of_quavers = templateMatchNotes(snippet, Quaver_up_start_template , 0.7, Staff_Size2, Treble_Lines)
    width = int( Quaver_up_start_template.shape[1] * scaling )
    height = int ( Quaver_up_start_template.shape[0] * scaling)
    if( len( number_of_quavers[0] ) != 0 ):
        x_pos, y_pos = number_of_quavers[1][0], number_of_quavers[0][0]
        cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
        snippet = cv2.putText(snippet, 'Q', (x_pos, y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
        if( St == 1 ):
            timing_cell.append("ST")
        else:
            timing_cell.append("N")
        timing_cell.append(quaver_time)
        timing_list.append(timing_cell)
        continue

    number_of_quavers = templateMatchNotes(snippet, Quaver_dn_start_template , 0.7, Staff_Size2, Treble_Lines)
    width = int( Quaver_dn_start_template.shape[1] * scaling )
    height = int ( Quaver_dn_start_template.shape[0] * scaling)
    if( len( number_of_quavers[0] ) != 0 ):
        x_pos, y_pos = number_of_quavers[1][0], number_of_quavers[0][0]
        cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
        snippet = cv2.putText(snippet, 'Q', (x_pos, y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
        if( St == 1 ):
            timing_cell.append("ST")
        else:
            timing_cell.append("N")
        timing_cell.append(quaver_time)
        timing_list.append(timing_cell)
        continue

    number_of_quavers = templateMatchNotes(snippet, Quaver_up_mid_template , 0.7, Staff_Size2, Treble_Lines)
    width = int( Quaver_up_mid_template.shape[1] * scaling )
    height = int ( Quaver_up_mid_template.shape[0] * scaling)
    if( len( number_of_quavers[0] ) != 0 ):
        x_pos, y_pos = number_of_quavers[1][0], number_of_quavers[0][0]
        cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
        snippet = cv2.putText(snippet, 'Q', (x_pos, y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
        if( St == 1 ):
            timing_cell.append("ST")
        else:
            timing_cell.append("N")
        timing_cell.append(quaver_time)
        timing_list.append(timing_cell)
        continue

    number_of_quavers = templateMatchNotes(snippet, Quaver_dn_mid_template , 0.7, Staff_Size2, Treble_Lines)
    width = int( Quaver_dn_mid_template.shape[1] * scaling )
    height = int ( Quaver_dn_mid_template.shape[0] * scaling)
    if( len( number_of_quavers[0] ) != 0 ):
        x_pos, y_pos = number_of_quavers[1][0], number_of_quavers[0][0]
        cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
        snippet = cv2.putText(snippet, 'Q', (x_pos, y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
        if( St == 1 ):
            timing_cell.append("ST")
        else:
            timing_cell.append("N")
        timing_cell.append(quaver_time)
        timing_list.append(timing_cell)
        continue

    number_of_quavers = templateMatchNotes(snippet, Quaver_up_end_template , 0.7, Staff_Size2, Treble_Lines)
    width = int( Quaver_up_end_template.shape[1] * scaling )
    height = int ( Quaver_up_end_template.shape[0] * scaling)
    if( len( number_of_quavers[0] ) != 0 ):
        x_pos, y_pos = number_of_quavers[1][0], number_of_quavers[0][0]
        cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
        snippet = cv2.putText(snippet, 'Q', (x_pos, y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
        if( St == 1 ):
            timing_cell.append("ST")
        else:
            timing_cell.append("N")
        timing_cell.append(quaver_time)
        timing_list.append(timing_cell)
        continue

    number_of_quavers = templateMatchNotes(snippet, Quaver_dn_end_template , 0.7, Staff_Size2, Treble_Lines)
    width = int( Quaver_dn_end_template.shape[1] * scaling )
    height = int ( Quaver_dn_end_template.shape[0] * scaling)
    if( len( number_of_quavers[0] ) != 0 ):
        x_pos, y_pos = number_of_quavers[1][0], number_of_quavers[0][0]
        cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
        snippet = cv2.putText(snippet, 'Q', (x_pos, y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
        if( St == 1 ):
            timing_cell.append("ST")
        else:
            timing_cell.append("N")
        timing_cell.append(quaver_time)
        timing_list.append(timing_cell)
        continue

    number_of_quavers = templateMatchNotes(snippet, Quaver_dn_single , 0.7, Staff_Size2, Treble_Lines)
    width = int( Quaver_up_end_template.shape[1] * scaling )
    height = int ( Quaver_up_end_template.shape[0] * scaling)
    if( len( number_of_quavers[0] ) != 0 ):
        x_pos, y_pos = number_of_quavers[1][0], number_of_quavers[0][0]
        cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
        snippet = cv2.putText(snippet, 'Q', (x_pos, y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
        if( St == 1 ):
            timing_cell.append("ST")
        else:
            timing_cell.append("N")
        timing_cell.append(quaver_time)
        timing_list.append(timing_cell)
        continue

    number_of_quavers = templateMatchNotes(snippet, Quaver_up_single , 0.7, Staff_Size2, Treble_Lines)
    width = int( Quaver_dn_end_template.shape[1] * scaling )
    height = int ( Quaver_dn_end_template.shape[0] * scaling)
    if( len( number_of_quavers[0] ) != 0 ):
        x_pos, y_pos = number_of_quavers[1][0], number_of_quavers[0][0]
        cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
        snippet = cv2.putText(snippet, 'Q', (x_pos, y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
        if( St == 1 ):
            timing_cell.append("ST")
        else:
            timing_cell.append("N")
        timing_cell.append(quaver_time)
        timing_list.append(timing_cell)
        continue
    
    
    number_of_quavers = templateMatchNotes(snippet, Quaver_slope_start_template , 0.7, Staff_Size2, Treble_Lines)
    width = int( Quaver_dn_end_template.shape[1] * scaling )
    height = int ( Quaver_dn_end_template.shape[0] * scaling)
    if( len( number_of_quavers[0] ) != 0 ):
        x_pos, y_pos = number_of_quavers[1][0], number_of_quavers[0][0]
        cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
        snippet = cv2.putText(snippet, 'Q', (x_pos, y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
        if( St == 1 ):
            timing_cell.append("ST")
        else:
            timing_cell.append("N")
        timing_cell.append(quaver_time)
        timing_list.append(timing_cell)
        continue
        
    number_of_quavers = templateMatchNotes(snippet, Quaver_slope_end_template , 0.7, Staff_Size2, Treble_Lines)
    width = int( Quaver_dn_end_template.shape[1] * scaling )
    height = int ( Quaver_dn_end_template.shape[0] * scaling)
    if( len( number_of_quavers[0] ) != 0 ):
        x_pos, y_pos = number_of_quavers[1][0], number_of_quavers[0][0]
        cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
        snippet = cv2.putText(snippet, 'Q', (x_pos, y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
        if( St == 1 ):
            timing_cell.append("ST")
        else:
            timing_cell.append("N")
        timing_cell.append(quaver_time)
        timing_list.append(timing_cell)
        continue


    
    #Crotchets-----------------------------------------------------------------------------------------------------------------------------------
    
    number_of_crotchets = templateMatchNotes(snippet, crotchets_up_template, 0.7, Staff_Size2, Treble_Lines)   
    if( len( number_of_crotchets[0] ) != 0 ):
        number_of_quavers = templateMatchNotes(snippet, Linelees_s_quaver_up_start_template , 0.75, Staff_Size2, Treble_Lines)
        width = int( Quaver_up_start_template.shape[1] * scaling )
        height = int ( Quaver_up_start_template.shape[0] * scaling)
        if( len( number_of_quavers[0] ) != 0 ):
            x_pos, y_pos = number_of_quavers[1][0], number_of_quavers[0][0]
            cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
            snippet = cv2.putText(snippet, 'S', (x_pos , y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
            if( St == 1 ):
                timing_cell.append("ST")
            else:
                timing_cell.append("N")
            timing_cell.append(semiquaver_time)
            timing_list.append(timing_cell)
            continue

        number_of_quavers = templateMatchNotes(snippet, Linelees_s_quaver_dn_start_template , 0.75, Staff_Size2, Treble_Lines)
        width = int( Quaver_dn_start_template.shape[1] * scaling )
        height = int ( Quaver_dn_start_template.shape[0] * scaling)
        if( len( number_of_quavers[0] ) != 0 ):
            x_pos, y_pos = number_of_quavers[1][0], number_of_quavers[0][0]
            cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
            snippet = cv2.putText(snippet, 'S', (x_pos, y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
            if( St == 1 ):
                timing_cell.append("ST")
            else:
                timing_cell.append("N")
            timing_cell.append(semiquaver_time)
            timing_list.append(timing_cell)
            continue

        number_of_quavers = templateMatchNotes(snippet, Linelees_s_quaver_up_mid_template , 0.75, Staff_Size2, Treble_Lines)
        width = int( Quaver_up_mid_template.shape[1] * scaling )
        height = int ( Quaver_up_mid_template.shape[0] * scaling)
        if( len( number_of_quavers[0] ) != 0 ):
            x_pos, y_pos = number_of_quavers[1][0], number_of_quavers[0][0]
            cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
            snippet = cv2.putText(snippet, 'S', (x_pos, y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
            if( St == 1 ):
                timing_cell.append("ST")
            else:
                timing_cell.append("N")
            timing_cell.append(semiquaver_time)
            timing_list.append(timing_cell)
            continue

        number_of_quavers = templateMatchNotes(snippet, Linelees_s_quaver_dn_mid_template , 0.75, Staff_Size2, Treble_Lines)
        width = int( Quaver_dn_mid_template.shape[1] * scaling )
        height = int ( Quaver_dn_mid_template.shape[0] * scaling)
        if( len( number_of_quavers[0] ) != 0 ):
            x_pos, y_pos = number_of_quavers[1][0], number_of_quavers[0][0]
            cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
            snippet = cv2.putText(snippet, 'S', (x_pos, y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
            if( St == 1 ):
                timing_cell.append("ST")
            else:
                timing_cell.append("N")
            timing_cell.append(semiquaver_time)
            timing_list.append(timing_cell)
            continue

        number_of_quavers = templateMatchNotes(snippet, Linelees_s_quaver_up_end_template , 0.75, Staff_Size2, Treble_Lines)
        width = int( Quaver_up_end_template.shape[1] * scaling )
        height = int ( Quaver_up_end_template.shape[0] * scaling)
        if( len( number_of_quavers[0] ) != 0 ):
            x_pos, y_pos = number_of_quavers[1][0], number_of_quavers[0][0]
            cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
            snippet = cv2.putText(snippet, 'S', (x_pos, y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
            if( St == 1 ):
                timing_cell.append("ST")
            else:
                timing_cell.append("N")
            timing_cell.append(semiquaver_time)
            timing_list.append(timing_cell)
            continue

        number_of_quavers = templateMatchNotes(snippet, Linelees_s_quaver_dn_end_template , 0.75, Staff_Size2, Treble_Lines)
        width = int( Quaver_dn_end_template.shape[1] * scaling )
        height = int ( Quaver_dn_end_template.shape[0] * scaling)
        if( len( number_of_quavers[0] ) != 0 ):
            x_pos, y_pos = number_of_quavers[1][0], number_of_quavers[0][0]
            cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
            snippet = cv2.putText(snippet, 'S', (x_pos, y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
            if( St == 1 ):
                timing_cell.append("ST")
            else:
                timing_cell.append("N")
            timing_cell.append(semiquaver_time)
            timing_list.append(timing_cell)
            continue
    
        
        x_pos, y_pos = number_of_crotchets[1][0], number_of_crotchets[0][0]
        cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
        snippet = cv2.putText(snippet, 'C', (x_pos, y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
        if( St == 1 ):
            timing_cell.append("ST")
        else:
            timing_cell.append("N")
        timing_cell.append(crotchet_time)
        timing_list.append(timing_cell)
        continue
        
    number_of_crotchets = templateMatchNotes(snippet, crotchets_dn_template, 0.7, Staff_Size2, Treble_Lines) 
    if( len( number_of_crotchets[0] ) != 0 ):
        
        number_of_quavers = templateMatchNotes(snippet, Linelees_s_quaver_up_start_template , 0.75, Staff_Size2, Treble_Lines)
        width = int( Quaver_up_start_template.shape[1] * scaling )
        height = int ( Quaver_up_start_template.shape[0] * scaling)
        if( len( number_of_quavers[0] ) != 0 ):
            x_pos, y_pos = number_of_quavers[1][0], number_of_quavers[0][0]
            cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
            snippet = cv2.putText(snippet, 'S', (x_pos , y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
            if( St == 1 ):
                timing_cell.append("ST")
            else:
                timing_cell.append("N")
            timing_cell.append(semiquaver_time)
            timing_list.append(timing_cell)
            continue

        number_of_quavers = templateMatchNotes(snippet, Linelees_s_quaver_dn_start_template , 0.75, Staff_Size2, Treble_Lines)
        width = int( Quaver_dn_start_template.shape[1] * scaling )
        height = int ( Quaver_dn_start_template.shape[0] * scaling)
        if( len( number_of_quavers[0] ) != 0 ):
            x_pos, y_pos = number_of_quavers[1][0], number_of_quavers[0][0]
            cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
            snippet = cv2.putText(snippet, 'S', (x_pos, y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
            if( St == 1 ):
                timing_cell.append("ST")
            else:
                timing_cell.append("N")
            timing_cell.append(semiquaver_time)
            timing_list.append(timing_cell)
            continue

        number_of_quavers = templateMatchNotes(snippet, Linelees_s_quaver_up_mid_template , 0.75, Staff_Size2, Treble_Lines)
        width = int( Quaver_up_mid_template.shape[1] * scaling )
        height = int ( Quaver_up_mid_template.shape[0] * scaling)
        if( len( number_of_quavers[0] ) != 0 ):
            x_pos, y_pos = number_of_quavers[1][0], number_of_quavers[0][0]
            cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
            snippet = cv2.putText(snippet, 'S', (x_pos, y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
            if( St == 1 ):
                timing_cell.append("ST")
            else:
                timing_cell.append("N")
            timing_cell.append(semiquaver_time)
            timing_list.append(timing_cell)
            continue

        number_of_quavers = templateMatchNotes(snippet, Linelees_s_quaver_dn_mid_template , 0.75, Staff_Size2, Treble_Lines)
        width = int( Quaver_dn_mid_template.shape[1] * scaling )
        height = int ( Quaver_dn_mid_template.shape[0] * scaling)
        if( len( number_of_quavers[0] ) != 0 ):
            x_pos, y_pos = number_of_quavers[1][0], number_of_quavers[0][0]
            cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
            snippet = cv2.putText(snippet, 'S', (x_pos, y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
            if( St == 1 ):
                timing_cell.append("ST")
            else:
                timing_cell.append("N")
            timing_cell.append(semiquaver_time)
            timing_list.append(timing_cell)
            continue

        number_of_quavers = templateMatchNotes(snippet, Linelees_s_quaver_up_end_template , 0.75, Staff_Size2, Treble_Lines)
        width = int( Quaver_up_end_template.shape[1] * scaling )
        height = int ( Quaver_up_end_template.shape[0] * scaling)
        if( len( number_of_quavers[0] ) != 0 ):
            x_pos, y_pos = number_of_quavers[1][0], number_of_quavers[0][0]
            cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
            snippet = cv2.putText(snippet, 'S', (x_pos, y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
            if( St == 1 ):
                timing_cell.append("ST")
            else:
                timing_cell.append("N")
            timing_cell.append(semiquaver_time)
            timing_list.append(timing_cell)
            continue

        number_of_quavers = templateMatchNotes(snippet, Linelees_s_quaver_dn_end_template , 0.75, Staff_Size2, Treble_Lines)
        width = int( Quaver_dn_end_template.shape[1] * scaling )
        height = int ( Quaver_dn_end_template.shape[0] * scaling)
        if( len( number_of_quavers[0] ) != 0 ):
            x_pos, y_pos = number_of_quavers[1][0], number_of_quavers[0][0]
            cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
            snippet = cv2.putText(snippet, 'S', (x_pos, y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
            if( St == 1 ):
                timing_cell.append("ST")
            else:
                timing_cell.append("N")
            timing_cell.append(semiquaver_time)
            timing_list.append(timing_cell)
            continue

    
        x_pos, y_pos = number_of_crotchets[1][0], number_of_crotchets[0][0]
        cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
        snippet = cv2.putText(snippet, 'C', (x_pos, y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
        if( St == 1 ):
            timing_cell.append("ST")
        else:
            timing_cell.append("N")
        timing_cell.append(crotchet_time)
        timing_list.append(timing_cell)
        continue

    
    number_of_quavers = templateMatchNotes(snippet, Lineless_quaver_up_start_template , 0.75, Staff_Size2, Treble_Lines)
    width = int( Quaver_up_start_template.shape[1] * scaling )
    height = int ( Quaver_up_start_template.shape[0] * scaling)
    if( len( number_of_quavers[0] ) != 0 ):
        x_pos, y_pos = number_of_quavers[1][0], number_of_quavers[0][0]
        cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
        snippet = cv2.putText(snippet, 'Q', (x_pos, y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
        if( St == 1 ):
            timing_cell.append("ST")
        else:
            timing_cell.append("N")
        timing_cell.append(quaver_time)
        timing_list.append(timing_cell)
        continue

    number_of_quavers = templateMatchNotes(snippet, Lineless_quaver_dn_start_template , 0.75, Staff_Size2, Treble_Lines)
    width = int( Quaver_dn_start_template.shape[1] * scaling )
    height = int ( Quaver_dn_start_template.shape[0] * scaling)
    if( len( number_of_quavers[0] ) != 0 ):
        x_pos, y_pos = number_of_quavers[1][0], number_of_quavers[0][0]
        cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
        snippet = cv2.putText(snippet, 'Q', (x_pos, y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
        if( St == 1 ):
            timing_cell.append("ST")
        else:
            timing_cell.append("N")
        timing_cell.append(quaver_time)
        timing_list.append(timing_cell)
        continue
    
    number_of_quavers = templateMatchNotes(snippet, Lineless_quaver_up_mid_template , 0.75, Staff_Size2, Treble_Lines)
    width = int( Quaver_up_mid_template.shape[1] * scaling )
    height = int ( Quaver_up_mid_template.shape[0] * scaling)
    if( len( number_of_quavers[0] ) != 0 ):
        x_pos, y_pos = number_of_quavers[1][0], number_of_quavers[0][0]
        cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
        snippet = cv2.putText(snippet, 'Q', (x_pos, y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
        if( St == 1 ):
            timing_cell.append("ST")
        else:
            timing_cell.append("N")
        timing_cell.append(quaver_time)
        timing_list.append(timing_cell)
        continue

    number_of_quavers = templateMatchNotes(snippet, Lineless_quaver_dn_mid_template , 0.75, Staff_Size2, Treble_Lines)
    width = int( Quaver_dn_mid_template.shape[1] * scaling )
    height = int ( Quaver_dn_mid_template.shape[0] * scaling)
    if( len( number_of_quavers[0] ) != 0 ):
        x_pos, y_pos = number_of_quavers[1][0], number_of_quavers[0][0]
        cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
        snippet = cv2.putText(snippet, 'Q', (x_pos, y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
        if( St == 1 ):
            timing_cell.append("ST")
        else:
            timing_cell.append("N")
        timing_cell.append(quaver_time)
        timing_list.append(timing_cell)
        continue
    
    number_of_quavers = templateMatchNotes(snippet, Lineless_quaver_up_end_template , 0.75, Staff_Size2, Treble_Lines)
    width = int( Quaver_up_end_template.shape[1] * scaling )
    height = int ( Quaver_up_end_template.shape[0] * scaling)
    if( len( number_of_quavers[0] ) != 0 ):
        x_pos, y_pos = number_of_quavers[1][0], number_of_quavers[0][0]
        cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
        snippet = cv2.putText(snippet, 'Q', (x_pos, y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
        if( St == 1 ):
            timing_cell.append("ST")
        else:
            timing_cell.append("N")
        timing_cell.append(quaver_time)
        timing_list.append(timing_cell)
        continue

    number_of_quavers = templateMatchNotes(snippet, Lineless_quaver_dn_end_template , 0.75, Staff_Size2, Treble_Lines)
    width = int( Quaver_dn_end_template.shape[1] * scaling )
    height = int ( Quaver_dn_end_template.shape[0] * scaling)
    if( len( number_of_quavers[0] ) != 0 ):
        x_pos, y_pos = number_of_quavers[1][0], number_of_quavers[0][0]
        cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
        snippet = cv2.putText(snippet, 'Q', (x_pos, y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
        if( St == 1 ):
            timing_cell.append("ST")
        else:
            timing_cell.append("N")
        timing_cell.append(quaver_time)
        timing_list.append(timing_cell)
        continue
    
    #Minims--------------------------------------------------------------------------------------------------------------------------------------
    number_of_crotchets = templateMatchNotes(snippet, minims_up_template, 0.7, Staff_Size2, Treble_Lines)   
    if( len( number_of_crotchets[0] ) != 0 ):
        x_pos, y_pos = number_of_crotchets[1][0], number_of_crotchets[0][0]
        cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
        snippet = cv2.putText(snippet, 'M', (x_pos, y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
        if( St == 1 ):
            timing_cell.append("ST")
        else:
            timing_cell.append("N")
        timing_cell.append(minin_time)
        timing_list.append(timing_cell)
        continue
    
    number_of_crotchets = templateMatchNotes(snippet, minims_dn_template, 0.7, Staff_Size2, Treble_Lines)   
    if( len( number_of_crotchets[0] ) != 0 ):
        x_pos, y_pos = number_of_crotchets[1][0], number_of_crotchets[0][0]
        cv2.rectangle(snippet, ( x_pos, y_pos ), ( x_pos + width, y_pos + height ), (0, 255, 255), 2)
        snippet = cv2.putText(snippet, 'M', (x_pos, y_pos + height + 15), font, fontScale, 0, 1, cv2.LINE_AA)
        if( St == 1 ):
            timing_cell.append("ST")
        else:
            timing_cell.append("N")
        timing_cell.append(minin_time)
        timing_list.append(timing_cell)
        continue
    

    timing_cell.append("N")
    temp = timing_list[len(timing_list) - 1]
    timing_cell.append(temp[1])
    timing_list.append(timing_cell)
    
        


"""

Count = 0
for snippet in snip_list:
    cv2.imwrite('Output' + str(Count) + '.png', snippet)
    Count = Count + 1

"""


clefType = "C"
if clefType == "G":
    keyboard = ["B-3", "C-4", "D-4", "E-4", "F-4", "G-4", "A-4", "B-4", "C-5", "D-5", "E-5", "F-5", "G-5", "A-5", "B-5", "C-6", "D-6"]
    temp = np.array(note_list)
    temp += 7
    for i in range(len(note_list)):
        note_list[i] = int(Note(keyboard[temp[i]]))

elif clefType == "C":
    keyboard = ["C-3", "D-3", "E-3", "F-3", "G-3", "A-3", "B-3", "C-4", "D-4", "E-4", "F-4", "G-4", "A-4", "B-4", "C-5", "D-5", "E-5"]
    temp = np.array(note_list)
    temp += 7
    for i in range(len(note_list)):
        note_list[i] = int(Note(keyboard[temp[i]]))
        print(">> #" + str(i) + ": " + keyboard[temp[i]])

elif clefType == "F":
    keyboard = ["D-2", "E-2", "F-2", "G-2", "A-2", "B-2", "C-3", "D-3", "E-3", "F-3", "G-3", "A-4", "B-4", "C-5", "D-5", "E-5", "F-5"]
    temp = np.array(note_list)
    temp += 7
    for i in range(len(note_list)):
        note_list[i] = int(Note(keyboard[temp[i]]))



for i in range(len( accidental_list ) ):
    if( accidental_list[ i ] == 'S' ):
        note_list[ i ] += 1

    elif( accidental_list[ i ] == 'F' ):
        note_list[ i ] -= 1



#----------------------------------------------------

print(">> Note list")
print(note_list)




BPM = 155
beat_time = 60 / BPM

#timing_list = np.zeros_like(note_list, dtype="float")
#timing_list = timing_list + beat_time

MIDI_list = list()
aux_index = 0
for i in range( len( timing_list ) ):
    temp = list()
    if( timing_list[ i ][ 0 ] == 'N' ):
        temp.append('N')
        temp.append( note_list[ aux_index ] )
        temp.append( timing_list[ i ][ 1 ] * beat_time)
        aux_index += 1
    elif( timing_list[ i ][ 0 ] == 'ST' ):
        temp.append('ST')
        temp.append( note_list[ aux_index ] )
        temp.append( timing_list[ i ][ 1 ] * beat_time)
        aux_index += 1
    else:
        temp.append( timing_list[ i ][ 0 ] )
        temp.append( timing_list[ i ][ 0 ] )
        temp.append( timing_list[ i ][ 1 ] * beat_time)
    
    
    MIDI_list.append(temp)




#------------------------MUSIC PLAYER-------------------------------
pygame.midi.init()
player = pygame.midi.Output(0)
player.set_instrument(0)

for i in MIDI_list:
    if( i[ 0 ] == 'P' ):
        time.sleep( i[ 2 ] )
        
    elif( i[ 0 ] == 'ST' ):
        player.note_on(i[1], 127)
        time.sleep( ( i[2] * 0.3 ) )
        player.note_off(i[1], 127)
        time.sleep( ( i[2] * 0.7 ) )
        
    else:
        player.note_on(i[1], 127)
        time.sleep(i[2])
        player.note_off(i[1], 127)
    
del player
pygame.midi.quit()
                                                                                        


plt.subplot( 2, 2, 1)
plt.imshow(raw_img, 'gray')
plt.title('Original')
plt.xticks([])
plt.yticks([])

plt.subplot( 2, 2, 2)
plt.imshow(workable_img, 'gray')
plt.title('Workable Image')
plt.xticks([])
plt.yticks([])

plt.subplot( 2, 2, 3)
plt.imshow(lineless_sheet, 'gray')
plt.title('Lineless notes')
plt.xticks([])
plt.yticks([])

plt.subplot( 2, 2, 4)
plt.imshow(copy_raw_img)
plt.title('Notes only')
plt.xticks([])
plt.yticks([])

plt.tight_layout()
plt.show()