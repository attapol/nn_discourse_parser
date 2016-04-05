"""Label functions

Each function should take a data_reader.DRelation object as an argument 
and output a label string.

"""
import json
import os

class LabelFunction(object):
    
    def label_name(self):
        raise NotImplementedError("Subclasses should implement this!")

    def label(self, drelation):
        raise NotImplementedError("Subclasses should implement this!")

    def valid_labels(self):
        raise NotImplementedError("Subclasses should implement this!")

class OriginalLabel(LabelFunction):

    def __init__(self):
        self.valid_label_set = set()

    def label_name(self):
        return 'original_label'

    def label(self, drelation):
        senses = drelation.senses
        sense = senses[0]
        self.valid_label_set.add(sense)
        return sense

    def valid_labels(self):
        return self.valid_label_set

class TopLevelLabel(LabelFunction):

    def label_name(self):
        return 'top_level_label'

    def label(self, drelation):
        senses = drelation.senses
        
        return senses[0].split('.')[0]

    def valid_labels(self):
        senses = [
                "Comparison",
                "Contingency",
                "Expansion",
                "Temporal",
        ]
        return senses

class PitlerLabel(LabelFunction):
    """ Labeling function before splitting into 4 binary classification

    EntRel will also be mixed into Expansion
    """

    def label_name(self):
        return 'entrel_mixed_in_top_label_label'

    def label(self, drelation):
        if drelation.relation_type == 'EntRel':
            return 'Expansion'
        elif drelation.relation_type == 'Implicit':
            senses = drelation.senses
            return senses[0].split('.')[0]
        else:
            return None

    def valid_labels(self):
        senses = [
                "Comparison",
                "Contingency",
                "Expansion",
                "Temporal",
        ]
        return senses

class SecondLevelLabel(LabelFunction):

    def label_name(self):
        return 'second_level_label'

    def label(self, drelation):
        split_sense = drelation.senses[0].split('.')
        num_split = len(split_sense)
        if num_split > 1 and not split_sense[1] == 'Condition' and \
                not split_sense[1] == 'Pragmatic condition' and \
                not split_sense[1] == 'Pragmatic contrast' and \
                not split_sense[1] == 'Pragmatic concession' and \
                not split_sense[1] == 'Exception':
            return split_sense[0]+"."+split_sense[1];
        return None

    def valid_labels(self):
        LEVEL2_SENSES = [
                "Comparison.Concession",
                "Comparison.Contrast",
                "Contingency.Cause",
                "Contingency.Pragmatic cause",
                "Expansion.Alternative",
                "Expansion.Conjunction",
                "Expansion.Instantiation",
                "Expansion.List",
                "Expansion.Restatement",
                "Temporal.Asynchronous",
                "Temporal.Synchrony"
        ]
        return LEVEL2_SENSES


"""Generic Mapper from JSON file

The mapper file should be in this format
{
    "label1" : {"dimension1": l1, "dimension2": l2 ...}
    "label2" : {"dimension1": l1, "dimension2": l2 ...}
    ...
    "labeln" : {...
    ...
}

"""
class GenericMapping(object):

    def __init__(self, json_file, exclude_na=False):
        self.mapping = json.load(open(json_file))
        base_name = os.path.basename(json_file)
        self.mapping_name = os.path.splitext(base_name)[0]

        self.dimension_set = self.validate_dimension_set(self.mapping)
        self.dimension_to_lf = {}
        for dimension in self.dimension_set:
            self.dimension_to_lf[dimension] = GenericMapping.JSONLabel(self.mapping, self.mapping_name, dimension, exclude_na)

    def get_label_function(self, dimension):
        return self.dimension_to_lf[dimension]

    def get_all_label_functions(self):
        return [self.dimension_to_lf[x] for x in self.dimension_to_lf]

    def validate_dimension_set(self, mapping):
        dimension_set = set()
        for sense in mapping:
            if len(dimension_set) > 0:
                new_dimension_set = set(mapping[sense].keys())
                assert (len(new_dimension_set) > 0 )
                assert (new_dimension_set == dimension_set)
                dimension_set = new_dimension_set
            else:
                dimension_set = set(mapping[sense].keys())
        return dimension_set


    class JSONLabel(LabelFunction):

        def __init__(self, mapping, mapping_name, dimension, exclude_na):
            self.mapping = mapping
            self.mapping_name = mapping_name
            self.dimension = dimension
            self.exclude_na = exclude_na

        def label_name(self):
            return '%s.%s' % (self.mapping_name, self.dimension)

        def label(self, drelation):
            senses = drelation.senses
            if senses[0] not in self.mapping:
                print '%s NOT FOUND! Skipping' % senses[0]
                return None
            else:
                label = self.mapping[senses[0]][self.dimension]
                if self.exclude_na and label == 'n.a.':
                    return None
                else: 
                    return label

        def valid_labels(self):
            label_set = set()
            for sense in self.mapping:
                label_set.add(self.mapping[sense][self.dimension])
            return label_set

class ComparisonBinary(LabelFunction):

    def label_name(self):
        return 'comparison_binary'

    def label(self, drelation):
        senses = drelation.senses
        s = senses[0].split('.')[0]
        if s == 'Comparison':
            return 'Comparison'
        else:
            return 'negative'

