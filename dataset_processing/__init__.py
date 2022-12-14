def get_dataset(dataset_name):
    dataset_name = str(dataset_name).lower()
    if dataset_name == 'best':
        from .best import WholeSlideImageDataset
        return WholeSlideImageDataset
    elif dataset_name == 'delta':
        #To be implemented
        from .delta import WholeSlideImageDataset
        return WholeSlideImageDataset
    else:
        raise NotImplementedError(f'Dataset Type {dataset_name} is Not Implemented')

def class_parser(stain='he', dysplasia_combine = True, respiratory_combine = True, gastric_combine = True, atypia_combine = True, p53_combine = True):
    '''
    stain - 'he' or 'p53'
    dysplasia_combine - combine the atypia of uncertain significance and dysplasia classes
    respiratory_combine - combine the respiratory mucosa cilia and respiratory mucosa classes
    gastric_combine - combine the tickled up columnar and gastric cardia classes
    atypia_combine - perform the following class mergers: atypia of uncertain significance+dysplasia, respiratory mucosa cilia+respiratory mucosa, tickled up columnar+gastric cardia classes, artifact+other
    p53_combine - perform the following class mergers: aberrant_positive_columnar, artifact+nonspecific_background+oral_bacteria, ignore equivocal_columnar
    '''
    
    class_names = ['artifact', 'background', 'immune_cells', 'squamous_mucosa']
    he_classes = ['gastric_cardia', 'intestinal_metaplasia', 'respiratory_mucosa']
    p53_classes = ['aberrant_positive_columnar', 'wild_type_columnar']

    if stain == "he":
        class_names.extend(he_classes)
        if atypia_combine:
            class_names.extend(['atypia'])
        else:
            if dysplasia_combine:
                class_names.extend(['other'])
                if respiratory_combine:
                    if not gastric_combine:
                        class_names.extend(['tickled_up_columnar', 'atypia'])
                else:
                    class_names.extend('respiratory_mucosa_cilia')
                    if gastric_combine:
                        class_names.extend(['atypia'])
                    else:
                        class_names.extend(['tickled_up_columnar', 'atypia'])

            else:
                class_names.extend([['atypia_of_uncertain_significance', 'other', 'dysplasia']])
                if not respiratory_combine:
                    class_names.extend('respiratory_mucosa_cilia')
                if not gastric_combine:
                        class_names.extend(['tickled_up_columnar'])
    
    else:
        class_names.extend(p53_classes)
        if not p53_combine:
            class_names.extend(['equivocal_columnar', 'nonspecific_background', 'oral_bacteria', 'respiratory_mucosa'])

    class_names.sort()
    return class_names