import { useCallback, useState } from "react";
import ExplainabilityPanel from "./components/ExplainabilityPanel";
import LiveAlertFeed from "./components/LiveAlertFeed";
import LoginForm from "./components/LoginForm";
import SummaryView from "./components/SummaryView";
import TriageView from "./components/TriageView";
import type { Credentials } from "./api";
import type { Alert, Tier2Explanation, Tier2ExplanationBroadcast } from "./types";

type Tab = "feed" | "triage" | "summary";

export default function App() {
  const [creds, setCreds] = useState<Credentials | null>(null);
  const [tab, setTab] = useState<Tab>("feed");
  const [selectedAlert, setSelectedAlert] = useState<Alert | null>(null);
  // Keyed by alert_id -- explanations received live over the WebSocket
  // while on the "Live feed" tab (the only tab that holds the connection
  // open, same as live alerts themselves). ExplainabilityPanel also
  // fetches on demand via REST, so this is a supplement, not the only path.
  const [liveExplanations, setLiveExplanations] = useState<Record<string, Tier2Explanation>>({});

  const handleExplanation = useCallback((explanation: Tier2ExplanationBroadcast) => {
    setLiveExplanations((prev) => ({ ...prev, [explanation.alert_id]: explanation }));
  }, []);

  if (!creds) {
    return <LoginForm onAuthenticated={setCreds} />;
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>IDS Alert Dashboard</h1>
        <nav className="tab-nav">
          <button className={tab === "feed" ? "active" : ""} onClick={() => setTab("feed")}>
            Live feed
          </button>
          <button className={tab === "triage" ? "active" : ""} onClick={() => setTab("triage")}>
            Triage
          </button>
          <button className={tab === "summary" ? "active" : ""} onClick={() => setTab("summary")}>
            Summary
          </button>
        </nav>
        <button className="signout-btn" onClick={() => setCreds(null)}>
          Sign out
        </button>
      </header>

      <div className="app-body">
        <main className="app-main">
          {tab === "feed" && <LiveAlertFeed creds={creds} onSelect={setSelectedAlert} onExplanation={handleExplanation} />}
          {tab === "triage" && <TriageView creds={creds} onSelect={setSelectedAlert} />}
          {tab === "summary" && <SummaryView creds={creds} />}
        </main>
        <ExplainabilityPanel
          alert={selectedAlert}
          creds={creds}
          liveExplanation={selectedAlert ? liveExplanations[selectedAlert.alert_id] : undefined}
          onClose={() => setSelectedAlert(null)}
        />
      </div>
    </div>
  );
}
