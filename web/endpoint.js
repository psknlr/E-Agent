/* Resolve a provider's base URL to the chat endpoint it implies. */
(function (scope) {
  'use strict';

  // The usual OpenAI-style "base URL" ends in a bare version segment such as
  // /v1 (https://api.minimax.cn/v1) or /v4. No provider's working chat
  // endpoint ends that way, so a URL like that cannot be correct as typed and
  // the request path it stands for is unambiguous. Anything else is treated
  // as the full endpoint and left exactly as the user wrote it.
  const VERSION_SEGMENT = /\/v\d+$/i;
  const SUFFIX = {openai_chat: '/chat/completions', anthropic_messages: '/messages'};

  // `href` must already be a valid absolute URL. Returns the URL to call and
  // whether it was completed, so a caller can show the user what it will use.
  function resolve(href, protocol) {
    const url = new URL(href);
    const suffix = Object.prototype.hasOwnProperty.call(SUFFIX, protocol) ? SUFFIX[protocol] : '';
    const path = url.pathname.replace(/\/+$/, '');
    if (!suffix || !VERSION_SEGMENT.test(path)) return {url: url.href, completed: false, appended: ''};
    url.pathname = path + suffix;
    return {url: url.href, completed: true, appended: suffix};
  }

  const api = {resolve};
  scope.EAgentEndpoint = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : window);
