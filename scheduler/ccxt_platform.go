package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// ccxtPlatform describes a ccxt-adapter exchange (currently okx, bybit) whose
// dispatcher path shares the OKX-structured scripts: check_<x>.py carries the
// signal/execute contract, fetch_<x>_positions/balance and close_<x>_position
// carry the live-ops wrappers. Live position-sync/kill-switch/close wiring is
// per-platform (OKX is fully wired; Bybit live is refused in config
// validation until its circuit/kill-switch path lands).
type ccxtPlatform struct {
	name            string // canonical platform string ("okx", "bybit")
	label           string // display label
	spotTakerFee    float64
	perpsTakerFee   float64
	envKey          string
	envSecret       string
	envPassphrase   string // okx only; empty when not required
	scriptPrefix    string // "okx" | "bybit" — derives script paths below
	checkScript     string
	positionsScript string
	balanceScript   string
	closeScript     string
	tvPrefix        string // "OKX" | "BYBIT" — TradingView vendor root
	liveWired       bool   // Go live machinery (positions fetch, CB, kill switch) exists
}

var ccxtPlatforms = map[string]*ccxtPlatform{
	"okx": {
		name:            "okx",
		label:           "OKX",
		spotTakerFee:    OKXSpotTakerFeePct,
		perpsTakerFee:   OKXPerpsTakerFeePct,
		envKey:          "OKX_API_KEY",
		envSecret:       "OKX_API_SECRET",
		envPassphrase:   "OKX_PASSPHRASE",
		scriptPrefix:    "okx",
		checkScript:     "shared_scripts/check_okx.py",
		positionsScript: "shared_scripts/fetch_okx_positions.py",
		balanceScript:   "shared_scripts/fetch_okx_balance.py",
		closeScript:     "shared_scripts/close_okx_position.py",
		tvPrefix:        "OKX",
		liveWired:       true,
	},
	"bybit": {
		name:            "bybit",
		label:           "Bybit",
		spotTakerFee:    BybitSpotTakerFeePct,
		perpsTakerFee:   BybitPerpsTakerFeePct,
		envKey:          "BYBIT_API_KEY",
		envSecret:       "BYBIT_API_SECRET",
		envPassphrase:   "",
		scriptPrefix:    "bybit",
		checkScript:     "shared_scripts/check_bybit.py",
		positionsScript: "shared_scripts/fetch_bybit_positions.py",
		balanceScript:   "shared_scripts/fetch_bybit_balance.py",
		closeScript:     "shared_scripts/close_bybit_position.py",
		tvPrefix:        "BYBIT",
		liveWired:       false,
	},
}

func ccxtPlatformFor(platform string) *ccxtPlatform {
	return ccxtPlatforms[strings.ToLower(strings.TrimSpace(platform))]
}

// isCCXTPerpStrategy reports a perps strategy on any ccxt-adapter platform.
func isCCXTPerpStrategy(sc StrategyConfig) bool {
	return sc.Type == "perps" && ccxtPlatformFor(sc.Platform) != nil
}

// isCCXTStrategy reports any strategy (spot or perps) on a ccxt-adapter
// platform whose check-script carries the OKX argv/JSON contract.
func isCCXTStrategy(sc StrategyConfig) bool {
	return ccxtPlatformFor(sc.Platform) != nil
}

// ccxtIsLive reports live intent for a ccxt-platform strategy. Platforms
// whose Go-side live machinery is not wired yet (bybit until its
// position-sync/circuit/kill-switch path lands) are reported not-live even
// if the args say --mode=live; config validation is the loud refusal, this
// is the silent runtime backstop.
func ccxtIsLive(sc StrategyConfig) bool {
	p := ccxtPlatformFor(sc.Platform)
	if p == nil || !p.liveWired {
		return false
	}
	return okxIsLive(sc.Args)
}

// isCCXTPerpState is the StrategyState twin of isCCXTPerpStrategy.
func isCCXTPerpState(s *StrategyState) bool {
	return s != nil && s.Type == "perps" && ccxtPlatformFor(s.Platform) != nil
}

// ccxtSymbol extracts the coin symbol from ccxt check-script argv
// (args[1]), identical shape to okxSymbol.
func ccxtSymbol(args []string) string { return okxSymbol(args) }

// ccxtInstType normalizes the perps inst-type arg per platform: OKX uses
// "swap", Bybit's linear perpetuals use "linear". Falls back to the
// platform default when the arg is absent.
func ccxtInstType(sc StrategyConfig) string {
	raw := okxInstType(sc.Args) // returns "swap" default / literal --inst-type
	p := ccxtPlatformFor(sc.Platform)
	if p == nil {
		return raw
	}
	if raw == "swap" && p.name == "bybit" {
		return "linear"
	}
	if raw == "linear" && p.name == "okx" {
		return "swap"
	}
	return raw
}

// ccxtFeePlatformKey maps a strategy to the CalculatePlatformSpotFee key for
// its fee table (spot vs perps variants).
func ccxtFeePlatformKey(s *StrategyState) string {
	if s == nil {
		return ""
	}
	p := ccxtPlatformFor(s.Platform)
	if p == nil {
		return s.Platform
	}
	if s.Type == "perps" {
		return p.name + "-perps"
	}
	return p.name
}

// ccxtPerpsMidsBase fetches public perp tickers and returns coin -> last price
// via the provider ticker endpoint. The Bybit path mirrors fetchOKXPerpsMids.
var bybitMainnetURL = "https://api.bybit.com"

func fetchBybitPerpsMids(coins []string) (map[string]float64, error) {
	if len(coins) == 0 {
		return map[string]float64{}, nil
	}
	url := bybitMainnetURL + "/v5/market/tickers?category=linear"
	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return nil, fmt.Errorf("http request: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("http %d from %s", resp.StatusCode, url)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read tickers response: %w", err)
	}
	var env struct {
		RetCode int    `json:"retCode"`
		RetMsg  string `json:"retMsg"`
		Result  struct {
			List []struct {
				Symbol string `json:"symbol"`
				Last   string `json:"lastPrice"`
			} `json:"list"`
		} `json:"result"`
	}
	if err := json.Unmarshal(body, &env); err != nil {
		return nil, fmt.Errorf("parse tickers response: %w", err)
	}
	if env.RetCode != 0 {
		return nil, fmt.Errorf("bybit api error code=%d msg=%s", env.RetCode, env.RetMsg)
	}
	want := make(map[string]string, len(coins))
	for _, c := range coins {
		want[c+"USDT"] = c
	}
	marks := make(map[string]float64, len(coins))
	for _, t := range env.Result.List {
		coin, ok := want[t.Symbol]
		if !ok {
			continue
		}
		p, err := strconv.ParseFloat(t.Last, 64)
		if err != nil || p <= 0 {
			continue
		}
		marks[coin] = p
	}
	return marks, nil
}
