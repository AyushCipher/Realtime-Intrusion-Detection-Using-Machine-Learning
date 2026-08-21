import { useState } from "react";
import { verifyCredentials, type Credentials } from "../api";

export default function LoginForm({ onAuthenticated }: { onAuthenticated: (creds: Credentials) => void }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setChecking(true);
    const creds = { username, password };
    try {
      const ok = await verifyCredentials(creds);
      if (ok) {
        onAuthenticated(creds);
      } else {
        setError("Invalid username or password.");
      }
    } catch {
      setError("Could not reach the dashboard API. Is it running?");
    } finally {
      setChecking(false);
    }
  }

  return (
    <div className="login-screen">
      <form className="login-card" onSubmit={handleSubmit}>
        <h1>IDS Alert Dashboard</h1>
        <p className="login-subtitle">Sign in with the dashboard API's Basic-auth credentials.</p>
        <label>
          Username
          <input value={username} onChange={(e) => setUsername(e.target.value)} autoFocus required />
        </label>
        <label>
          Password
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required />
        </label>
        {error && <div className="login-error">{error}</div>}
        <button type="submit" disabled={checking}>
          {checking ? "Signing in..." : "Sign in"}
        </button>
      </form>
    </div>
  );
}
