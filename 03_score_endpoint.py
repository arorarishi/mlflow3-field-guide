"""Call the REST endpoint started by `mlflow models serve`, then show schema enforcement."""
import json
import urllib.error
import urllib.request

from common import load_splits

URL = "http://127.0.0.1:5001/invocations"


def post(payload):
    request = urllib.request.Request(
        URL, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


def main():
    _, _, X_test, _, _, y_test = load_splits()
    rows = X_test.head(3)
    status, body = post({"dataframe_split": rows.to_dict(orient="split")})
    print(f"Valid request -> HTTP {status}: {body}")
    print(f"Actual labels: {y_test.head(3).tolist()}")

    broken = rows.drop(columns=["mean radius"])
    status, body = post({"dataframe_split": broken.to_dict(orient="split")})
    print(f"Missing column -> HTTP {status}: {body[:220]}")


if __name__ == "__main__":
    main()
