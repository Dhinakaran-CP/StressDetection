import React, { useState } from "react";
import { API_BASE } from "../config";

export default function StressChatbot({ stressLevel, stressPercentage, open, onClose }) {
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
    <div className="fixed right-6 bottom-6 z-50">
      {open && (
        <div className="w-[360px] h-[500px] bg-surface-container-lowest border border-outline-variant/20 rounded-2xl flex flex-col shadow-lg overflow-hidden animate-in fade-in slide-in-from-bottom-4 duration-300">
          <div className="flex items-center justify-between p-4 bg-surface-container-low border-b border-outline-variant/10">
            <strong className="font-headline-sm text-lg text-primary">Stress Assistant</strong>
            <button className="text-on-surface-variant hover:text-primary transition-colors text-xs font-bold font-label-caps tracking-wider" onClick={onClose}>
              CLOSE
            </button>
          </div>

          {provider === "local-fallback" && (
            <div className="p-3 bg-amber-50 border-b border-amber-200 text-amber-800 text-xs">
              Gemini not connected. Using local fallback responses. Set GEMINI_API_KEY in .env to enable Gemini 2.5 Flash.
            </div>
          )}

          <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-4 bg-surface">
            {messages.map((msg, idx) => (
              <div
                key={`${msg.role}-${idx}`}
                className={`p-3 rounded-2xl max-w-[85%] text-[13px] leading-relaxed ${
                  msg.role === "user"
                    ? "self-end bg-primary text-on-primary rounded-br-sm shadow-sm"
                    : "self-start bg-white border border-outline-variant/15 text-on-surface shadow-sm rounded-bl-sm"
                }`}
              >
                {renderMessage(msg.text)}
              </div>
            ))}
            {loading && <div className="self-start bg-white border border-outline-variant/15 text-on-surface-variant p-3 rounded-2xl rounded-bl-sm text-sm shadow-sm italic">Thinking...</div>}
          </div>

          <div className="p-4 bg-surface-container-lowest border-t border-outline-variant/10 flex gap-2">
            <input
              className="flex-1 bg-surface-container-low border border-outline-variant/30 rounded-xl px-4 py-2.5 outline-none focus:ring-2 focus:ring-primary text-xs text-on-surface placeholder:text-on-surface-variant/60 font-medium"
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
            <button className="bg-primary text-on-primary px-4 py-2.5 rounded-xl text-xs font-bold font-label-caps tracking-wider shadow hover:opacity-95 active:scale-95 transition-all disabled:opacity-50" onClick={sendMessage} disabled={loading || !input.trim()}>
              Send
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
