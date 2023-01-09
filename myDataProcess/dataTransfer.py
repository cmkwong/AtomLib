from pyts.image import GramianAngularField

# encoding time series as images
def timeSeries2Img(timeSeries, method='summation'):
    """
    timeSeries: pd.Series
    """
    gaf = GramianAngularField(method=method)
    X_gaf = gaf.fit_transform(timeSeries)
    return X_gaf