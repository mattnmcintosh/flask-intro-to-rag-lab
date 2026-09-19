from __future__ import annotations

from flask import Flask, jsonify, request

from ai_client import generate_response
from company_documents import COMPANY_DOCUMENTS
from rag_service import build_prompt, retrieve_context, source_metadata


def create_app():
    app = Flask(__name__)

    @app.get("/api/health")
    def health_check():
        return jsonify({"status": "ok"})

    @app.post("/api/ask")
    def ask_question():
        """Accept a query and return a source-backed generated answer.

        TODO:
        1. Read JSON request data safely.
        2. Validate that `query` is a non-empty string.
        3. Retrieve relevant context from COMPANY_DOCUMENTS.
        4. If no context is found, return a safe fallback with an empty sources list.
        5. Build a structured prompt from the selected context.
        6. Call generate_response(prompt).
        7. Return query, answer, and sources as JSON.
        8. If generate_response raises RuntimeError, return a 503 service error.
        """
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.get_json()
        if not data or "query" not in data:
            return jsonify({"error": "Missing query field"}), 400

        query = data["query"]
        if not isinstance(query, str) or not query.strip():
            return jsonify({"error": "Query must be a non-empty string"}), 400

        # Retrieve relevant context
        matches = retrieve_context(query, COMPANY_DOCUMENTS)

        # If no context is found, return a safe fallback with an empty sources list
        if not matches:
            fallback_answer = (
                "The provided company documents do not contain enough information "
                "to answer your question."
            )
            return jsonify({
                "query": query,
                "answer": fallback_answer,
                "sources": [],
            }), 200

        # Build prompt from context
        prompt = build_prompt(query, matches)

        try:
            answer = generate_response(prompt)
        except RuntimeError:
            return jsonify({"error": "Model service failed to respond."}), 503

        sources = [source_metadata(match) for match in matches]

        return jsonify({
            "query": query,
            "answer": answer,
            "sources": sources,
        }), 200

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
