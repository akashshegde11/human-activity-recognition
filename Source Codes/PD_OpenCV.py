# system libraries
import os
import shutil
import time
import cv2


# natural sorting using regular expressions
import re
_nsre = re.compile('([0-9]+)')


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]


trg_data_root = "D:\Documents\FrameConversion\KTH\\"

class_labels = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]
frame_path = "\\frames\\"
frame_set_prefix = "person"
person_count = 25

rec_prefix = "d"
rec_count = 4
seg_prefix = "seg"
seg_count = 4

time_start = time.time()
for x in range(0, len(class_labels)):

    class_folder = trg_data_root + class_labels[x]
    class_frame_path = trg_data_root + class_labels[x] + frame_path
    class_frame_path_cmd = "mkdir " + class_frame_path
    os.system(class_frame_path_cmd)

indices_file = open("D:\Documents\FrameConversion\KTH\\00sequences.txt", "r")
for j in range(1, person_count+1):
    person_name = indices_file.readline()

    print("person ", j)

    for i in range(0, len(class_labels)):
        print(class_labels[i], "\t")
        class_folder = trg_data_root + class_labels[i]
        class_frame_path = trg_data_root + class_labels[i] + frame_path
        if j < 10:
            person_prefix = "person0" + str(j) + "_" + class_labels[i] + "_"
        else:
            person_prefix = "person" + str(j) + "_" + class_labels[i] + "_"
        for k in range(1,rec_count+1):
            rec_filename = class_folder + "\\" + person_prefix + "d" + str(k) + "_uncomp.avi"
            person_subfolder = person_prefix + "d" + str(k)
            print(person_subfolder)
            output_folder = class_frame_path + person_subfolder
            output_folder_cmd = "mkdir " + output_folder
            recording_name = indices_file.readline()
            print(recording_name)
            seg_line = indices_file.readline()
            print(seg_line)
            if recording_name.rstrip() == person_subfolder and j > 0:
                print(k)
                # ffmpeg_cmd = "ffmpeg -i " + rec_filename + " " + output_folder + "\\frame%d.jpg"
                os.system(output_folder_cmd)
                # os.system(ffmpeg_cmd)
                cap = cv2.VideoCapture(rec_filename)
                video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                print("Number of frames: ", video_length)
                count_cap = 0
                print("Converting video...")
                while cap.isOpened():
                    ret, frame = cap.read()
                    cv2.imwrite(output_folder + "\\frame%d.jpg" % (count_cap+1), frame)
                    count_cap += 1
                    if count_cap > (video_length-1):
                        cap.release()
                        print("%d frames extracted" % count_cap)
                        break
                segments = seg_line.split(', ')
                for p in range(0, seg_count):
                    seg_name = seg_prefix + str(p+1)
                    seg_folder = output_folder + "\\" + seg_name
                    seg_folder_cmd = "mkdir " + seg_folder
                    os.system(seg_folder_cmd)
                    seg_string = segments[p].rstrip()
                    start_and_finish = seg_string.split('-')
                    for q in range(int(start_and_finish[0]), int(start_and_finish[1])+1):
                        frame_name = "frame" + str(q) + ".jpg"
                        source_frame = output_folder + "\\" + frame_name
                        shutil.move(source_frame, seg_folder)

            else:
                print(rec_filename)
                print("Invalid index: skipping this recording ")
        print("")

indices_file.close()
time_end = time.time()
print("Time taken for conversion of videos to frames: ", time_end - time_start, " seconds")
