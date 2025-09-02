import { useState, useEffect, useRef, useContext } from "react";
import { useTask } from "@/context/TaskContext";
import { useAuth } from "@/context/AuthContext";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Send } from "lucide-react";
import AuthContext from "../context/AuthContext";
import { useForm } from "react-hook-form";
import AxiosInstance from "./utils/AxiosInstance";
import { toast } from "react-toastify";

const CommentSection = ({ taskId }) => {
  const { getTaskComments, addComment } = useTask();
  const { user } = useContext(AuthContext);
  const [comments, setComments] = useState([]);
  const [newComment, setNewComment] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [mentionedUsers, setMentionedUsers] = useState([]);
  const inputRef = useRef(null);

  // Form default values
  const defaultValues = {
    task: taskId,
    author: user.user_id, // current loggedIn user
    content: "",
    mentions: [],
  };

  // react-hook-form hooks
  const {
    handleSubmit,
    register,
    setValue,
    formState: { errors },
  } = useForm({
    defaultValues: defaultValues,
  });

  const createComment = async (data) => {
    setSubmitting(true);

    try {
      const res = await AxiosInstance.post(`comments/`, {
        task: taskId,
        author: user.user_id,
        content: newComment,
        mentions: data.mentions,
      });
      if (res.status === 201 || res.status === 200) {
        toast.success("Comment Added!");
        // Add new comment immediately
        const newCommentObject = {
          id: res.data.id,
          author: { username: user.username },
          content: res.data.content,
          created_at: res.data.created_at || new Date().toISOString(), // fallback if not returned
        };
        setComments((prevComments) => [...prevComments, newCommentObject]);
        setNewComment("");
        setValue("content", ""); // Reset react-hook-form input
      } else {
        toast.error("Failed to add comment.");
      }
    } catch (e) {
      toast.error(e.message);
    } finally {
      setSubmitting(false);
    }
  };

  useEffect(() => {
    // Fetch comments for this task
    const taskComments = getTaskComments(taskId);
    setComments(taskComments);
  }, [taskId, getTaskComments]);

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return new Intl.DateTimeFormat("en-US", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    }).format(date);
  };

  // Parse the text and render mentions with special styling
  const renderCommentWithMentions = (text) => {
    const mentionRegex = /@(\w+\s\w+)/g;
    const parts = text.split(mentionRegex);

    return parts.map((part, index) => {
      if (index % 2 === 1) {
        // This is a mention
        return (
          <span key={index} className="mention">
            @{part}
          </span>
        );
      }
      return <span key={index}>{part}</span>;
    });
  };

  return (
    <div className="px-6 pb-4">
      <div className="mb-3 max-h-60 overflow-y-auto">
        {comments.length > 0 ? (
          comments.map((comment) => (
            <div key={comment.id} className="mb-3">
              <div className="flex items-start space-x-2">
                <Avatar className="h-8 w-8">
                  <AvatarFallback>
                    {comment.author.username.charAt(0)}
                  </AvatarFallback>
                </Avatar>
                <div className="flex-1">
                  <div className="bg-gray-50 rounded-lg p-3">
                    <div className="flex justify-between mb-1">
                      <span
                        className="font-medium text-sm"
                        style={{ fontWeight: "bold" }}
                      >
                        {comment.author.username}
                      </span>
                      <span className="text-xs text-gray-500">
                        {formatDate(comment.created_at)}
                      </span>
                    </div>
                    <p className="text-sm">
                      {renderCommentWithMentions(comment.content)}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          ))
        ) : (
          <p className="text-center text-sm text-gray-500">No comments yet</p>
        )}
      </div>

      <form onSubmit={handleSubmit(createComment)} className="flex space-x-2">
        <Input
          ref={inputRef}
          value={newComment}
          onChange={(e) => {
            setNewComment(e.target.value);
            setValue("content", e.target.value); // Update react-hook-form too
          }}
          placeholder="Add a comment..."
          disabled={submitting}
          className="flex-1"
        />
        <Button type="submit" size="sm" disabled={submitting}>
          <Send className="h-4 w-4" />
        </Button>
      </form>
    </div>
  );
};

export default CommentSection;
