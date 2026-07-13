import React, { useState } from "react";
import { API_BASE } from "../config";

export default function StressChatbot({ stressLevel, stressPercentage }) {
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [provider, setProvider] = useState("unknown");
  const [messages, setMessages] = useState([
    {
      role: "bot",
      text: "Hi, I am your stress support assistant. Ask me about stress, calm routines, focus breaks, or sleep hygiene.",
    },
  ]);

  const sendMessage = async () => {
    const text = input.trim();
    if (!text || loading) return;

    const nextMessages = [...messages, { role: "user", text }];
    setMessages(nextMessages);
    setInput("");
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE}/api/chat/stress`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          stress_level: stressLevel || "Unknown",
          stress_percentage:
            stressPercentage !== null && stressPercentage !== undefined
              ? Number(stressPercentage)
              : null,
        }),
      });

      const data = await response.json();
      if (data.status === "success" && data.reply) {
        setProvider(data.provider || "unknown");
        setMessages((prev) => [...prev, { role: "bot", text: data.reply }]);
      } else {
        setMessages((prev) => [
          ...prev,
          {
            role: "bot",
            text: data.message || "I could not process that right now. Please try again.",
          },
        ]);
      }
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "bot",
          text: `Network issue: ${err.message}. Please ensure backend is running.`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const renderMessage = (text) => {
    if (!text) return null;
    
    // 1. Escape raw HTML to prevent browser from swallowing <tags>
    let html = text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

    // 2. Replace **text** with <strong>text</strong>
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    // 3. Replace '* text' list items with a styled div
    html = html.replace(/(?:^|\s)\* (.*?)(?=(?:\s\* |$))/g, '<div style="margin-top: 6px; margin-left: 12px;">• $1</div>');
    // 4. Replace newlines with <br/>
    html = html.replace(/\n/g, '<br/>');
    
    return <span dangerouslySetInnerHTML={{ __html: html }} />;
  };

  return (
    <div className="stress-chatbot-shell">
      {open && (
        <div className="stress-chatbot-panel">
          <div className="stress-chatbot-header">
            <strong>Stress Assistant</strong>
            <button className="btn btn-secondary" onClick={() => setOpen(false)}>
              Close
            </button>
          </div>

          {provider === "local-fallback" && (
            <div className="chat-provider-warning">
              Gemini not connected. Using local fallback responses. Set GEMINI_API_KEY in .env to enable Gemini 2.5 Flash.
            </div>
          )}

          <div className="stress-chatbot-messages">
            {messages.map((msg, idx) => (
              <div
                key={`${msg.role}-${idx}`}
                className={msg.role === "user" ? "chat-msg-user" : "chat-msg-bot"}
              >
                {renderMessage(msg.text)}
              </div>
            ))}
            {loading && <div className="chat-msg-bot">Thinking...</div>}
          </div>

          <div className="stress-chatbot-input">
            <input
              className="form-control"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about stress relief..."
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  sendMessage();
                }
              }}
            />
            <button className="btn btn-primary" onClick={sendMessage} disabled={loading || !input.trim()}>
              Send
            </button>
          </div>
        </div>
      )}

      {!open && (
        <button className="btn btn-primary stress-chatbot-open" onClick={() => setOpen(true)}>
          Stress Chat
        </button>
      )}
    </div>
  );
}
