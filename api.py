from dataclasses import asdict

from flask import Flask, render_template, request
from werkzeug.exceptions import HTTPException

from constraints import check_feasibility
from optimizer import optimize_plan
from serialization import problem_from_dict, result_to_dict

app = Flask(__name__)
app.json.ensure_ascii = False


@app.get("/")
def index():
    return render_template("index.html")


@app.errorhandler(HTTPException)
def http_error(error: HTTPException):
    messages = {
        400: "Некорректный JSON или запрос.",
        404: "Адрес API не найден.",
        405: "Метод запроса не поддерживается для этого адреса.",
        415: "Укажите Content-Type: application/json.",
        500: "Внутренняя ошибка сервера.",
    }

    response = error.get_response()
    response.data = app.json.dumps({
        "status": "error" if error.code == 500 else "invalid_request",
        "message": messages.get(error.code, error.description),
    })
    response.content_type = "application/json"
    return response


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/validate")
def validate_problem():
    try:
        depots, stations, _ = problem_from_dict(request.get_json())
        report = check_feasibility(depots, stations)
    except ValueError as error:
        return {
            "status": "invalid_data",
            "message": str(error),
        }, 400

    return {
        "status": "feasible" if report.feasible else "infeasible",
        "feasible": report.feasible,
        "station_ids": list(report.station_ids),
        "minimum_delivery": report.minimum_delivery.tolist(),
        "available_capacity": report.available_capacity.tolist(),
        "total_stock": report.total_stock,
        "total_required": report.total_required,
        "violations": [asdict(item) for item in report.violations],
    }


@app.post("/api/optimize")
def calculate_plan():
    method = request.args.get("method", "highs")

    try:
        depots, stations, costs = problem_from_dict(request.get_json())
        result = optimize_plan(depots, stations, costs, method=method)
    except ValueError as error:
        return {
            "status": "invalid_data",
            "message": str(error),
        }, 400

    response = result_to_dict(result, depots, stations)
    http_status = 500 if result.status == "error" else 200
    return response, http_status


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)