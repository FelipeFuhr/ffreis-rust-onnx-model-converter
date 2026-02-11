def test_sklearn_convert(tmp_path):
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import load_iris
    import joblib

    X, y = load_iris(return_X_y=True)
    m = LogisticRegression().fit(X, y)

    p = tmp_path/"m.joblib"
    joblib.dump(m, p)

    from onnx_converter.converters.sklearn import SklearnConverter
    out = tmp_path/"m.onnx"

    SklearnConverter().convert(p, out, n_features=4)

    import onnxruntime as ort
    ort.InferenceSession(str(out))
