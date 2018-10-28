**i2b2 2014 Evaluation Script**

This script is distributed as a part of the i2b2 2014 Cardiac Risk and 
Protected Health Information (PHI) tasks. 

If you would like to contribute to this project, pull requests are welcome.
Please see: https://help.github.com/articles/fork-a-repo for instructions
on how to make a fork of this repository, and
https://help.github.com/articles/using-pull-requests for instructions on
making a pull request. Suggestions for improvements, bugs or feature requests
may be directed to the i2b2 evaluation scripts' issues page located at:
https://github.com/kotfic/i2b2_evaluation_scripts/issues

_Setup_

This script also requires the following Python packages:
lxml version 3.3.1
numpy version 1.8.0

If you get an error when running the script, please make sure that these
are installed and accessible to your Python installation.


_Running the script_

This script intended to be used via
command line:
python evaluate.py [cr|phi] [FLAGS] SYSTEM GOLD

Where 'cr' produces Precision, Recall and F1 (P/R/F1) measure for the 
cardiac risk task and 'phi' produces P/R/F1 for the PHI task. SYSTEM and GOLD 
may be individual files representing system output in the case of SYSTEM and
the gold standard in the case of GOLD.  SYSTEM and GOLD may also be 
directories in which case all files in SYSTEM will be compared to files the 
GOLD directory based on their file names.   See below for more information 
on the different output the cr/phi flag produces.



_File name restrictions_

File names MUST be of the form:
XXX-YY.xml where XXX is the patient id,  and YY is the document id. The 
files from your system runs are matched to the the gold standard file by 
file name alone.  If your system outputs file names in a different format, 
you will need to either modify your system or this script.


_Output for Risk Factor Track_

To compare your system output for the Risk Factor track, run the following 
command for individual files:

python evaluate.py cr {system.xml} {gold.xml}
(replace the file names in {}s with the names of your actual files)

or, to run the script on directories of files:
python evaluate.py cr {system}/ {gold}/
(again, replace the folder names in {}s with the names of your actual folders)

Running one of these versions will produce output in this format:

```
 (# of files)            Measure        Macro (SD)     Micro               
---------------------------------------------------------------------------
Total                    Precision      1.0 (0.0)      1.0                 
                         Recall         1.0 (0.0)      1.0                 
                         F1             1.0            1.0              
```

The script evaluates the accuracy of your tags based on tag type 
and all the attributes (except ID).  If you want to get more details 
about the output of your system, such as which attributes it is 
getting right/wrong, you can use the more experimental flags.  Please see 
the evaluate.py script itself for more information on the flags.


_Output for De-identification Track_

To compare your system output for the de-identification track, run the following 
command on individual files:

python evaluate.py phi {system.xml} {gold.xml}
(replace the file names in {}s with the names of your actual files)

or, to run the script on directories of files:
python evaluate.py phi {system}/ {gold}/
(again, replace the folder names in {}s with the names of your actual folders)


Running one of these versions wil produce output that looks like this:

```
Strict (521)             Measure        Macro (SD)     Micro               
---------------------------------------------------------------------------
Total                    Precision      0.6635 (0.11)  0.6537              
                         Recall         0.4906 (0.12)  0.4988              
                         F1             0.5641         0.5658              


Relaxed (521)            Measure        Macro (SD)     Micro               
---------------------------------------------------------------------------
Total                    Precision      0.897 (0.086)  0.9047              
                         Recall         0.6663 (0.15)  0.6903              
                         F1             0.7646         0.7831              


HIPAA Strict (521)       Measure        Macro (SD)     Micro               
---------------------------------------------------------------------------
Total                    Precision      0.7406 (0.098) 0.7225              
                         Recall         0.7406 (0.098) 0.7225              
                         F1             0.7406         0.7225              


HIPAA Relaxed (521)      Measure        Macro (SD)     Micro               
---------------------------------------------------------------------------
Total                    Precision      1.0 (0.0)      1.0                 
                         Recall         1.0 (0.0)      1.0                 
                         F1             1.0            1.0                 
```

A few notes to explain this output:
-  The "(521)" represents the number of files the scrip was run on
-  "Strict" evaluations require that the offsets for the system outputs match *exactly*
-  "Relaxed" evaluations allow for the "end" part of the offsets to be off by 2--this allows for variations in including "'s" and other endings that many systems will ignore due to tokenization
-  "HIPPA" evalutions include only the tags that a strict interpretation of the HIPAA guidelines.  See the below list for which tags are included in this evaluation



_HIPAA-compliant PHI_

- NAME/PATIENT
- AGE
- LOCATION/CITY
- LOCATION/STREET
- LOCATION/ZIP
- LOCATION/ORGANIZATION
- DATE
- CONTACT/PHONE
- CONTACT/FAX
- CONTACT/EMAIL
- ID/SSN
- ID/MEDICALRECORD
- ID/HEALTHPLAN
- ID/ACCOUNT
- ID/LICENSE
- ID/VEHICLE
- ID/DEVICE
- ID/BIOID
- ID/IDNUM 


_Verbose flag_

To get document-by-document information about the accuracy of your tags, you can use the
"-v" or "--verbose" flag.  For example:

python evaluate.py cr -v system/ gold/


_Advanced useage_

Some additional functionality is made available for testing and error 
analysis. This functionality is provided AS IS with the hopes that it will
be useful. It should be considered 'experimental' at best, may be bug prone
and will not be explicitly supported, though, bug reports and pull requests
are welcome.

Advanced Flags:

--filter [TAG ATTRIBUTES] :: run P/R/F1 measures in either summary or verbose
                             mode (see -v) for the list of attributes defined
                             by TAG ATTRIBUTES. This may be a comma separated
                             list of tag names and attribute values. For more
                             see Advanced Examples.
--conjunctive :: If multiple values are passed to filter as a comma separated
                 list, treat them as a series of AND based filters instead of
                 a series of OR based filters
--invert :: run P/R/F1 on the inverted set of tags defined by TAG ATTRIBUTES
            in the --filter tag (see --filter).

Advanced Examples:

python evaluate.py cr --filter MEDICATION system/ gold/ 

  Evaluate system output in system/ folder against gold/ folder considering
only MEDICATION tags

python evaluate.py cr --filter CAD,OBESE system/ gold/ 

  Evaluate system output in system/ folder against gold/ folder considering
only CAD or OBESE tags. Comma separated lists to the --filter flag are con-
joined via OR.

python evaluate.py cr --filter "CAD,before DCT" system/ gold/ 

  Evaluate system output in system/ folder against gold/ folder considering
only CAD *OR* tags with a time attribute of before DCT. This is probably 
not what you want when filtering, see the next example

python evaluate.py cr --conjunctive \
                         --filter "CAD,before DCT" system/ gold/ 

  Evaluate system output in system/ folder against gold/ folder considering
CAD tags *AND* tags with a time attribute of before DCT.

python evaluate.py cr --invert \
                         --filter MEDICATION system/ gold/

 Evaluate system output in system/ folder against gold/ folder considering
any tag which is NOT a MEDICATION tag.

python evaluate.py cr --invert \
                         --conjunctive \
                         --filter "CAD,before DCT" system/ gold/ 

 Evaluate system output in system/ folder against gold/ folder considering
any tag which is NOT CAD and with a time attribute of 'before DCT'
