# Modified by Max Friedrich, 2018

###############################################################################
#
#   Copyright 2014 Christopher Kotfila
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
#
# i2b2 2014 Evaluation Scripts
#
# This script is distributed as apart of the i2b2 2014 Cardiac Risk and
# Personal Health-care Information (PHI) tasks. It is intended to be used via
# command line:
#
# $> python evaluate.py [cr|phi] [FLAGS] SYSTEM GOLD
#
#   Where 'cr' produces Precision, Recall and F1 (P/R/F1) measure for the
# cardiac risk task and 'phi' produces P/R/F1 for the PHI task. SYSTEM and GOLD
# may be individual files representing system output in the case of SYSTEM and
# the gold standard in the case of GOLD.  SYSTEM and GOLD may also be
# directories in which case all files in SYSTEM will be compared to files the
# GOLD directory based on their file names. File names MUST be of the form:
# XXX-YY.xml where XXX is the patient id,  and YY is the document id. See the
# README.md file for more details.
#
# Basic Flags:
# -v, --verbose :: Print document by document P/R/F1 for each document instead
#                  of summary statistics for an entire set of documents.
#
# Basic Examples:
#
# $> python evaluate.py cr system.xml gold.xml
#
#   Evaluate the single system output file 'system.xml' against the gold
# standard file 'gold.xml' for the Cardiac Risk Task (the 'cr' argument).
# Please note:
# file to file comparisons are made available for testing purposes,  systems
# output will be evaluated using the "batch" system/ gold/ examples as shown
# below.
#
# $> python evaluate.py cr system/ gold/
#
#   Evaluate the set of system outputs in the folder system/ against the set of
# gold standard annotations in gold/ using the cardiac risk task evaluation.
#
# $> python evaluate.py phi system/ gold/
#
#   Evaluate the set of system outputs in the folder system against the set of
# gold standard annotations in gold/ using the PHI task evaluation.
#
#
#
# Advanced Usage:
#
#   Some additional functionality is made available for testing and error
# analysis. This functionality is provided AS IS with the hopes that it will
# be useful. It should be considered 'experimental' at best, may be bug prone
# and will not be explicitly supported, though, bug reports and pull requests
# are welcome.
#
# Advanced Flags:
#
# --filter [TAG ATTRIBUTES] :: run P/R/F1 measures in either summary or verbose
#                              mode (see -v) for the list of attributes defined
#                              by TAG ATTRIBUTES. This may be a comma separated
#                              list of tag names and attribute values. For more
#                              see Advanced Examples.
# --conjunctive :: If multiple values are passed to filter as a comma separated
#                  list, treat them as a series of AND based filters instead of
#                  a series of OR based filters
# --invert :: run P/R/F1 on the inverted set of tags defined by TAG ATTRIBUTES
#             in the --filter tag (see --filter).
#
# Advanced Examples:
#
# $> python evaluate.py cr --filter MEDICATION system/ gold/
#
#   Evaluate system output in system/ folder against gold/ folder considering
# only MEDICATION tags
#
# $> python evaluate.py cr --filter CAD,OBESE system/ gold/
#
#   Evaluate system output in system/ folder against gold/ folder considering
# only CAD or OBESE tags. Comma separated lists to the --filter flag are con-
# joined via OR.
#
# $> python evaluate.py cr --filter "CAD,before DCT" system/ gold/
#
#   Evaluate system output in system/ folder against gold/ folder considering
# only CAD *OR* tags with a time attribute of before DCT. This is probably
# not what you want when filtering, see the next example
#
# $> python evaluate.py cr --conjunctive \
#                          --filter "CAD,before DCT" system/ gold/
#
#   Evaluate system output in system/ folder against gold/ folder considering
# CAD tags *AND* tags with a time attribute of before DCT.
#
# $> python evaluate.py cr --invert \
#                          --filter MEDICATION system/ gold/
#
#  Evaluate system output in system/ folder against gold/ folder considering
# any tag which is NOT a MEDICATION tag.
#
# $> python evaluate.py cr --invert \
#                          --conjunctive \
#                          --filter "CAD,before DCT" system/ gold/
#
#  Evaluate system output in system/ folder against gold/ folder considering
# any tag which is NOT CAD and with a time attribute of 'before DCT'


import argparse
import os
from collections import defaultdict

from .classes import StandoffAnnotation, Evaluate, CombinedEvaluation, \
    PHITrackEvaluation, CardiacRiskTrackEvaluation
from .tags import DocumentTag, PHITag, MEDICAL_TAG_CLASSES


# This function is 'exterimental' as in it works for my use cases
# But is not generally well documented or a part of the expected
# workflow.
def get_predicate_function(arg, tag):
    """ This function takes a tag attribute value, determines the attribute(s)
    of the class(es) this value belongs to,  and then returns a predicate
    function that returns true if this value is set for the  calculated
    attribute(s) on the class(es). This allows for overlap - ie. "ACE
    Inhibitor" is a valid type1 and a valid type2 attribute value.  If arg
    equals "ACE Inhibitor" our returned predicate function will return true if
    our tag has "ACE Inhibitor" set for either type1 or type2 attributes.
    Currently this is implemented to ONLY work with MEDICAL_TAG_CLASSES but
    could be easily extended to work with PHI tag classes.
    """
    attrs = []

    # Get a list of valid attributes for this argument
    # If we have a tag name (ie. MEDICATION) add 'name' to the attributes
    if arg in list(tag.tag_types.keys()):
        attrs.append("name")
    else:
        tag_attributes = ["valid_type1", "valid_type2", "valid_indicator",
                          "valid_status", "valid_time", "valid_type"]
        for cls in MEDICAL_TAG_CLASSES:
            for attr in tag_attributes:
                try:
                    if arg in getattr(cls, attr):
                        # add the attribute,  strip out the "valid_" prefix
                        # This assumes that classes follow the
                        # valid_ATTRIBUTE convention
                        # and will break if they are extended
                        attrs.append(attr.replace("valid_", ""))
                except AttributeError:
                    continue
        # Delete these so we don't end up carrying around
        # references in our function
        try:
            del tag_attributes
            del cls
            del attr
        except NameError:
            pass

    attrs = list(set(attrs))

    if len(attrs) == 0:
        print(("WARNING: could not find valid class attribute for " +
               "\"{}\", + skipping.".format(arg)))
        return lambda t: True

    # Define the predicate function we will use. artrs are scoped into
    # the closure,  which is sort of the whole point of the
    # get_predicate_function function.
    def matchp(t):
        for attr in attrs:
            if attr == "name" and t.name == arg:
                return True
            else:
                try:
                    if getattr(t, attr).lower() == arg.lower():
                        return True
                except (AttributeError, KeyError):
                    pass
        return False

    return matchp


def get_document_dict_by_system_id(system_dirs):
    """Takes a list of directories and returns all of the StandoffAnnotation's
    as a system id, annotation id indexed dictionary. System id (or
    StandoffAnnotation.sys_id) is whatever values trail the XXX-YY file id.
    For example:
       301-01foo.xml
       patient id:   301
       document id:  01
       system id:    foo

    In the case where there is nothing trailing the document id,  the sys_id
    is the empty string ('').
    """
    documents = defaultdict(lambda: defaultdict(int))

    for d in system_dirs:
        for fn in [f for f in os.listdir(d) if f.endswith('.xml')]:
            sa = StandoffAnnotation(os.path.join(d, fn))
            documents[sa.sys_id][sa.id] = sa

    return documents


def evaluate(system, gs, eval_class, **kwargs):
    """Evaluate the system by calling the eval_class (either EvaluatePHI or
    EvaluateCardiacRisk classes) with an annotation id indexed dict of
    StandoffAnnotation classes for the system(s) and the gold standard outputs.
    'system' will be a list containing either one file,  or one or more
    directories. 'gs' will be a file or a directory.  This function mostly just
    handles formatting arguments for the eval_class.
    """
    assert issubclass(eval_class, Evaluate) or \
           issubclass(eval_class, CombinedEvaluation), \
        "Must pass in EvaluatePHI or EvaluateCardiacRisk classes to evaluate()"

    gold_sa = {}
    evaluations = []

    # Strip verbose keyword if it exists
    # verbose is not a keyword to our eval classes
    # __init__() functions
    try:
        verbose = kwargs['verbose']
        del kwargs['verbose']
    except KeyError:
        verbose = False

    assert os.path.exists(gs), "{} does not exist!".format(gs)

    for s in system:
        assert os.path.exists(s), "{} does not exist!".format(s)

    # Handle if two files were passed on the command line
    if os.path.isfile(system[0]) and os.path.isfile(gs):
        gs = StandoffAnnotation(gs)
        s = StandoffAnnotation(system[0])
        e = eval_class({s.id: s}, {gs.id: gs}, **kwargs)
        e.print_docs()
        evaluations.append(e)

    # Handle the case where 'gs' is a directory and 'system' is a
    # list of directories.  For individual evaluation (one system output
    #  against the gold standard) this is a little overkill,  but this
    # lets us run multiple systems against the gold standard and get numbers
    # for each system output. useful for annotator agreement and final system
    # evaluations. Error checking to ensure consistent files in each directory
    # will be handled by the evaluation class.
    elif all([os.path.isdir(s) for s in system]) and os.path.isdir(gs):
        # Get a dict of gold standoff annotation indexed by id
        for fn in [f for f in os.listdir(gs) if f.endswith('.xml')]:
            sa = StandoffAnnotation(os.path.join(gs, fn))
            gold_sa[sa.id] = sa

        for s_id, system_sa in list(get_document_dict_by_system_id(system).items()):
            e = eval_class(system_sa, gold_sa, **kwargs)
            e.print_report(verbose=verbose)
            evaluations.append(e)

    else:
        raise Exception("Must pass file.xml file.xml  or [directory/]+ directory/"
                        "on command line!")

    return evaluations[0] if len(evaluations) == 1 else evaluations


def main():
    parser = argparse.ArgumentParser(description="To Write")

    sp = parser.add_subparsers(dest="sp", help="To Write")

    sp_phi = sp.add_parser("phi",
                           help="convert a document to different types")

    sp_phi.add_argument('--filter',
                        help="Filters to apply, use with invert & conjunction")
    sp_phi.add_argument('--conjunctive',
                        help="if multiple filters are applied, should these be \
                        combined with 'and' or 'or'",
                        action="store_true")
    sp_phi.add_argument('--invert',
                        help="Invert the list of filters,  match only tags \
                        that do not match filter functions",
                        action="store_true")
    sp_phi.add_argument('-v', '--verbose',
                        help="list full document by document scores",
                        action="store_true")
    sp_phi.add_argument("from_dirs",
                        help="directories to pull documents from",
                        nargs="+")
    sp_phi.add_argument("to_dir",
                        help="directories to save documents to")

    sp_cr = sp.add_parser("cr",
                          help="convert a document to different types")

    sp_cr.add_argument('--filter',
                       help="Filters to apply,  use with invert & conjunction")
    sp_cr.add_argument('--conjunctive',
                       help="if multiple filters are applied, should these be \
                       combined with 'and' or 'or'",
                       action="store_true")
    sp_cr.add_argument('--invert',
                       help="Invert the list of filters,  match only tags \
                       that do not match filter functions",
                       action="store_true")
    sp_cr.add_argument('-v', '--verbose',
                       help="list full document by document scores",
                       action="store_true")
    sp_cr.add_argument("from_dirs",
                       help="directories to pull documents from",
                       nargs="+")
    sp_cr.add_argument("to_dir",
                       help="directories to save documents to")

    args = parser.parse_args()

    if args.filter:
        evaluate(args.from_dirs, args.to_dir,
                 PHITrackEvaluation if args.sp == "phi" else
                 CardiacRiskTrackEvaluation,
                 verbose=args.verbose,
                 invert=args.invert,
                 conjunctive=args.conjunctive,
                 filters=[get_predicate_function(a, PHITag if args.sp == "phi" else DocumentTag)
                          for a in args.filter.split(",")])
    else:
        evaluate(args.from_dirs, args.to_dir,
                 PHITrackEvaluation if args.sp == "phi" else
                 CardiacRiskTrackEvaluation,
                 verbose=args.verbose)


if __name__ == "__main__":
    main()
